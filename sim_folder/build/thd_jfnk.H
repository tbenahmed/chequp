#ifndef THD_JFNK_H
#define THD_JFNK_H
//
// Jacobian-Free Newton-Krylov solve of backward-Euler electron thermal
// diffusion for a two-temperature Castro setup.
//
//   F(E) = E - E^n - dt * div( kappa(Te) grad Te ) = 0
//
// E = UEINT is the TOTAL internal energy density. The heavy-species
// energy E_h is frozen across the sub-step -- electron conduction does
// not appear in the heavy energy equation -- so at every iterate
//
//   f = E_h / E,    UFX = rho * f,
//
// which is what makes the EOS return the electron temperature.
//
//
// THE LINEAR SYSTEM, AND WHY MLMG BELONGS IN IT
// ---------------------------------------------
// The Frechet derivative of F, with s = dTe/dE = Te/E_e (exact for a
// gamma-law electron gas), is
//
//   J v = v - dt*div( kappa grad(s v) ) - dt*div( v s kappa' grad Te ).
//
// Change variables to the TEMPERATURE perturbation w = s v. The first
// two terms become
//
//   M w = (E_e/Te) w - dt*div( kappa grad w ),
//
// which is exactly MLABecLaplacian with A = 1, alpha = E_e/Te, B = dt,
// beta = kappa on faces, and homogeneous Neumann/periodic BCs -- the
// same face coefficients the operator already builds. This is a shifted
// variable-coefficient Laplacian: MLMG solves it with an iteration
// count independent of dx, where point-Jacobi-preconditioned GMRES does
// not, since cond(J) ~ 1 + dt*chi/dx^2.
//
// The third term (kappa' = 2.5*kappa/Te for Spitzer, so comparable to or
// larger than the second at a steep front, and non-symmetric) cannot go
// into an MLABecLaplacian. That gives the two schemes below:
//
//   MODE_IMPLICIT  Exact Newton. The matvec stays matrix-free (finite
//                  differences on F, so ALL of J including kappa'), and
//                  M^{-1} -- a fixed number of MLMG V-cycles -- is used
//                  as the left preconditioner instead of the diagonal.
//                  Quadratic convergence, few Krylov iterations.
//
//   MODE_MG        Frozen-coefficient Newton. Drop the kappa' term, so
//                  J = M exactly and there is no Krylov layer at all:
//                  one MLMG solve per Newton iteration gives the
//                  direction directly. Linear convergence, but each
//                  iteration is much cheaper and there is no
//                  finite-difference step to tune.
//
// Both produce a direction dE = (E_e/Te) * w and then share the same
// line search, source back-out and cache, so they are directly
// comparable at runtime.
//
#include <thd_diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <vector>
#include <string>
#include <memory>

using namespace amrex;

namespace thd_jfnk {

//   MODE_OFF       contribute nothing. Use with castro.diffuse_temp = 1
//                  to run Castro's own explicit diffusion instead.
//   MODE_IMPLICIT  backward-Euler JFNK, exact Jacobian, MG-preconditioned.
//   MODE_MG        backward-Euler with frozen conductivity, MLMG only.

enum DiffMode { MODE_OFF = 0, MODE_IMPLICIT = 1, MODE_MG = 2 };

//   PC_JACOBI  exact diagonal of the discretized Jacobian. Kept as a
//              fallback and as an A/B baseline: it is the diagonal of
//              the same M that PC_MG inverts (see Precond::update), so
//              switching between them changes only the quality of the
//              approximation, never the equation being solved.
//   PC_MG      a fixed number of MLMG V-cycles on M.

enum PrecondType { PC_JACOBI = 0, PC_MG = 1 };

struct Params {
    DiffMode    mode    = MODE_IMPLICIT;
    PrecondType precond = PC_MG;

    // Castro evaluates ext_src more than once per step from the same
    // old-time state. Reuse the result when the input is identical;
    // guarded by a checksum, so a mismatch falls through to a real solve.
    bool use_cache = true;

    // Multiplies the returned source. Castro combines the old-time and
    // new-time evaluations; try 0.5 to test whether that over-applies.
    Real source_scale = 1.0_rt;

    // Newton. Tolerance is relative to the energy scale, not to the
    // initial residual, which would be self-referential.
    //
    // MODE_MG converges linearly, not quadratically, so it needs a
    // larger iteration budget than MODE_IMPLICIT for the same tolerance.
    int  max_newton     = 20;
    Real newton_rtol    = 1.0e-6_rt;
    int  max_linesearch = 8;
    Real fd_eps         = 1.0e-7_rt;   // finite-difference step for J*v

    // Krylov (MODE_IMPLICIT only). max_gmres = 0 auto-sizes from the
    // grid, capped at gmres_ceiling. With PC_MG the count should be
    // single digits and dx-independent, so the ceiling is a diagnostic
    // rather than a real limit: if it is ever reached, the warning
    // below means the preconditioner is not doing its job.
    int  max_gmres      = 0;
    int  gmres_ceiling  = 120;
    Real gmres_rtol     = 1.0e-4_rt;

    // Second Gram-Schmidt pass when orthogonalization collapses the
    // vector norm by more than this factor. 0 disables.
    Real reorth_tol = 0.7_rt;

    // MLMG.
    //
    // mg_fixed_iter: V-cycles per PRECONDITIONER application. GMRES
    //   requires a FIXED LINEAR operator, so this is a fixed count, not
    //   a tolerance -- a tolerance-based inner solve makes M^{-1}
    //   depend on its own right-hand side and the Arnoldi relation
    //   silently stops holding. Use FGMRES if you want that.
    // mg_bottom_smoother: for the same reason, the bottom solve defaults
    //   to plain smoothing rather than BiCGStab, whose iteration count
    //   depends on the right-hand side. Irrelevant in MODE_MG.
    // mg_rtol / mg_max_iter: MODE_MG only, where the MLMG solve IS the
    //   linear solve and a tolerance is the right control.
    int  mg_fixed_iter         = 2;
    bool mg_bottom_smoother    = true;
    Real mg_rtol               = 1.0e-6_rt;
    Real mg_atol               = 0.0_rt;
    int  mg_max_iter           = 100;
    int  mg_pre_smooth         = 2;
    int  mg_post_smooth        = 2;
    int  mg_max_coarsening_lev = 30;
    bool mg_agglomeration      = true;
    bool mg_consolidation      = true;
    int  mg_verbose            = 0;
    int  mg_bottom_verbose     = 0;

    // Per-step diagnostics. Costs two extra coefficient evaluations.
    bool verbose = false;
};


// Runtime parameters, read once from the inputs file.
//   exact Newton, MG-preconditioned
//                      thd_diffusion.mode    = implicit
//                      thd_diffusion.precond = mg
//                      castro.diffuse_temp   = 0
//   frozen-coefficient, MLMG only
//                      thd_diffusion.mode    = mg
//                      castro.diffuse_temp   = 0
//   old behaviour (diagonal preconditioner)
//                      thd_diffusion.mode    = implicit
//                      thd_diffusion.precond = jacobi
//   explicit baseline  thd_diffusion.mode    = off
//                      castro.diffuse_temp   = 1

inline Params& runtime_params ()
{
    static Params p;
    static bool initialised = false;
    if (initialised) { return p; }
    initialised = true;

    amrex::ParmParse pp("thd_diffusion");

    std::string m;
    if (pp.query("mode", m)) {
        if      (m == "off")      { p.mode = MODE_OFF; }
        else if (m == "implicit") { p.mode = MODE_IMPLICIT; }
        else if (m == "mg")       { p.mode = MODE_MG; }
        else { amrex::Abort("thd_diffusion.mode must be 'off', 'implicit' or 'mg'"); }
    }

    std::string pcs;
    if (pp.query("precond", pcs)) {
        if      (pcs == "jacobi") { p.precond = PC_JACOBI; }
        else if (pcs == "mg")     { p.precond = PC_MG; }
        else { amrex::Abort("thd_diffusion.precond must be 'jacobi' or 'mg'"); }
    }

    pp.query("source_scale",  p.source_scale);
    pp.query("max_newton",    p.max_newton);
    pp.query("newton_rtol",   p.newton_rtol);
    pp.query("max_gmres",     p.max_gmres);
    pp.query("gmres_ceiling", p.gmres_ceiling);
    pp.query("gmres_rtol",    p.gmres_rtol);
    pp.query("reorth_tol",    p.reorth_tol);
    pp.query("fd_eps",        p.fd_eps);

    pp.query("mg_fixed_iter",         p.mg_fixed_iter);
    pp.query("mg_rtol",               p.mg_rtol);
    pp.query("mg_atol",               p.mg_atol);
    pp.query("mg_max_iter",           p.mg_max_iter);
    pp.query("mg_pre_smooth",         p.mg_pre_smooth);
    pp.query("mg_post_smooth",        p.mg_post_smooth);
    pp.query("mg_max_coarsening_lev", p.mg_max_coarsening_lev);
    pp.query("mg_verbose",            p.mg_verbose);
    pp.query("mg_bottom_verbose",     p.mg_bottom_verbose);

    int itmp = p.verbose ? 1 : 0;
    if (pp.query("verbose", itmp))   { p.verbose = itmp; }
    itmp = p.use_cache ? 1 : 0;
    if (pp.query("use_cache", itmp)) { p.use_cache = itmp; }
    itmp = p.mg_agglomeration ? 1 : 0;
    if (pp.query("mg_agglomeration", itmp)) { p.mg_agglomeration = itmp; }
    itmp = p.mg_consolidation ? 1 : 0;
    if (pp.query("mg_consolidation", itmp)) { p.mg_consolidation = itmp; }
    itmp = p.mg_bottom_smoother ? 1 : 0;
    if (pp.query("mg_bottom_smoother", itmp)) { p.mg_bottom_smoother = itmp; }

    if (p.mg_fixed_iter < 1) {
        amrex::Abort("thd_diffusion.mg_fixed_iter must be >= 1: a "
                     "tolerance-controlled inner solve is not a fixed linear "
                     "preconditioner and breaks GMRES.");
    }

    // Castro's own temperature diffusion is EXPLICIT (it evaluates
    // div(kappa grad T) and adds it as a source, with estdt_temp_diffusion
    // limiting dt). Running it alongside this solver applies conduction
    // twice; running neither applies none. Both are silent, so catch them.
    int castro_diffuse_temp = 0;
    { amrex::ParmParse pc("castro"); pc.query("diffuse_temp", castro_diffuse_temp); }

    const bool we_diffuse = (p.mode != MODE_OFF);

    if (we_diffuse && castro_diffuse_temp == 1) {
        amrex::Abort("Conduction applied TWICE: thd_diffusion.mode != off and "
                     "castro.diffuse_temp = 1. Set one of them off.");
    }
    if (!we_diffuse && castro_diffuse_temp != 1) {
        amrex::Print() << "[THD-JFNK] WARNING: no thermal conduction anywhere "
                       << "(mode = off, castro.diffuse_temp != 1).\n";
    }

    if (p.verbose) {
        const char* mname = (p.mode == MODE_OFF) ? "off"
                          : (p.mode == MODE_MG)  ? "mg" : "implicit";
        amrex::Print() << "[THD-JFNK] mode = " << mname;
        if (p.mode == MODE_IMPLICIT) {
            amrex::Print() << ", precond = "
                           << ((p.precond == PC_MG) ? "mg" : "jacobi");
        }
        amrex::Print() << ", castro.diffuse_temp = " << castro_diffuse_temp << "\n";
    }
    return p;
}


// Ghost fill.
//
// FillBoundary covers same-level and periodic ghosts. Ghosts OUTSIDE the
// domain are then filled by zero-gradient extrapolation, a no-flux
// (adiabatic) thermal wall -- the same condition MLABecLaplacian is
// given below as homogeneous Neumann.
//
// Ghosts at a COARSE-FINE boundary are touched by neither, and are left
// at the FillPatch values the caller supplied.
//
// The extrapolation source is clamped to the fab's own valid box, not to
// the domain: under domain decomposition a box interior to the domain
// would otherwise index outside its own data.

inline void
fill_ghosts (MultiFab& state, const Geometry& geom, int scomp, int ncomp)
{
    state.FillBoundary(scomp, ncomp, geom.periodicity());

    const Box& domain = geom.Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);
    const int per[AMREX_SPACEDIM] = {
        AMREX_D_DECL(geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2))
    };

    for (MFIter mfi(state); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.fabbox();
        const auto vlo = amrex::lbound(mfi.validbox());
        const auto vhi = amrex::ubound(mfi.validbox());
        Array4<Real> const s = state.array(mfi);

        amrex::ParallelFor(gbx, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) noexcept
        {
            const int n = scomp + nn;
            bool outside = false;

            if (per[0] == 0 && (i < dlo.x || i > dhi.x)) { outside = true; }
#if AMREX_SPACEDIM >= 2
            if (per[1] == 0 && (j < dlo.y || j > dhi.y)) { outside = true; }
#endif
#if AMREX_SPACEDIM == 3
            if (per[2] == 0 && (k < dlo.z || k > dhi.z)) { outside = true; }
#endif
            if (outside) {
                const int ii = amrex::min(amrex::max(i, vlo.x), vhi.x);
#if AMREX_SPACEDIM >= 2
                const int jj = amrex::min(amrex::max(j, vlo.y), vhi.y);
#else
                const int jj = j;
#endif
#if AMREX_SPACEDIM == 3
                const int kk = amrex::min(amrex::max(k, vlo.z), vhi.z);
#else
                const int kk = k;
#endif
                s(i,j,k,n) = s(ii,jj,kk,n);
            }
        });
    }
}

inline void
fill_ghosts (MultiFab& state, const Geometry& geom)
{
    fill_ghosts(state, geom, 0, state.nComp());
}


// Refresh UFX = rho * E_h/E over valid + ghost cells, so the EOS sees an
// f consistent with the current energy iterate.
//
// f is bounded strictly inside (0,1): at f = 1 the electron energy is
// exactly zero and Te degenerate, which restarts the kappa ~ Te^(5/2)
// feedback loop.

inline void
refresh_aux (MultiFab& state, const MultiFab& Eh)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real>       const s  = state.array(mfi);
        Array4<Real const> const eh = Eh.const_array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real E   = s(i,j,k,UEINT);
            const Real rho = s(i,j,k,URHO);
            if (E > 0.0_rt && rho > 0.0_rt && amrex::Math::isfinite(eh(i,j,k))) {
                constexpr Real f_min = 1.0e-12_rt;
                constexpr Real f_max = 1.0_rt - 1.0e-8_rt;
                const Real f = amrex::min(amrex::max(eh(i,j,k) / E, f_min), f_max);
                s(i,j,k,UFX) = rho * f;
            }
        });
    }
}

// F(E) = E - E^n - dt*L(E). `state` needs filled ghosts and an aux
// consistent with its UEINT.
//
// The Laplacian scratch comes from the workspace: allocating it here
// would mean one MultiFab per residual, and a solve performs ~120.
inline void
residual (const MultiFab& state, const MultiFab& En, MultiFab& F,
          const Geometry& geom, Real dt, thd_diff::Workspace* ws)
{
    MultiFab& L = ws->lap;
    thd_diff::compute_L(state, L, geom, ws);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(F, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real const> const en = En.const_array(mfi);
        Array4<Real const> const l  = L.const_array(mfi);
        Array4<Real>       const f  = F.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            f(i,j,k) = s(i,j,k,UEINT) - en(i,j,k) - dt * l(i,j,k);
        });
    }
}

// UEINT = base + lambda*delta, then restore consistency.
//
// No clamping here: clamping inside the iteration makes Newton converge
// on a modified equation. Positivity is enforced by the line search.
//
// Only UEINT ghosts are exchanged. refresh_aux then derives UFX over the
// grown box, and rho/species/temperature are frozen for the whole solve.
inline void
apply_update (MultiFab& state, const MultiFab& base, const MultiFab& delta,
              Real lambda, const MultiFab& Eh, const Geometry& geom)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real>       const s = state.array(mfi);
        Array4<Real const> const b = base.const_array(mfi);
        Array4<Real const> const d = delta.const_array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            s(i,j,k,UEINT) = b(i,j,k) + lambda * d(i,j,k);
        });
    }
    fill_ghosts(state, geom, UEINT, 1);
    refresh_aux(state, Eh);
}

// Minimum electron energy E - E_h over the valid region, in one fused
// reduction. Doing it as copy + subtract + min would cost two kernels,
// a reduction and a MultiFab, up to max_linesearch times per Newton
// iteration.
//
// E > E_h also implies E > 0 (E_h >= 0 by construction), so no separate
// positivity reduction is needed.
inline Real
min_electron_energy (const MultiFab& state, const MultiFab& Eh)
{
    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Eh, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real const> const eh = Eh.const_array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return { s(i,j,k,UEINT) - eh(i,j,k) };
        });
    }

    Real emin = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealMin(emin);
    return emin;
}


// ====================================================================
// The shifted Laplacian M, used as a preconditioner (MODE_IMPLICIT) or
// as the linear solve itself (MODE_MG).
//
//   M w = alpha*w - dt*div( kappa grad w ),    alpha = E_e/Te = (dTe/dE)^-1
//
// and the direction in energy space is recovered as dE = alpha*w, since
// w = s*dE with s = 1/alpha.
//
// Consistency check on the two preconditioners: the diagonal of M in
// w-space is alpha + dt*stiff, so the diagonally-scaled version of this
// preconditioner returns
//
//   alpha*r/(alpha + dt*stiff) = r/(1 + dt*stiff*(Te/E_e)),
//
// which is exactly the PC_JACOBI divisor below. PC_MG is the same
// operator with the off-diagonal coupling retained, so it can only
// change the iteration count, never the fixed point.
//
// dt -> 0 sanity check: M -> alpha, w = r/alpha, dE = alpha*w = r, i.e.
// M^-1 -> I, matching J -> I.
// ====================================================================

struct Precond {

    PrecondType type = PC_MG;
    bool        defined = false;

    // PC_JACOBI
    MultiFab diag;

    // PC_MG / MODE_MG
    MultiFab alpha;                              // E_e/Te, cell-centred
    MultiFab wsol;                               // solution, 1 ghost
    MultiFab rhs;                                // right-hand side
    std::unique_ptr<MLABecLaplacian> op;
    std::unique_ptr<MLMG>            mlmg;
    Geometry geom;
    Real     dt_cur = 0.0_rt;
    Params   par;

    bool need_mg () const { return type == PC_MG; }

    // ---------------------------------------------------------------
    // Allocated once per solve. The MLABecLaplacian grid hierarchy is
    // built here and reused across Newton iterations; only the
    // coefficients change, via update() below.
    void define (const BoxArray& ba, const DistributionMapping& dm,
                 const Geometry& gm, const Params& p, bool mg_needed)
    {
        par    = p;
        geom   = gm;
        defined = true;

        if (!mg_needed) {
            diag.define(ba, dm, 1, 0);
            return;
        }

        alpha.define(ba, dm, 1, 0);
        wsol .define(ba, dm, 1, 1);
        rhs  .define(ba, dm, 1, 0);

        LPInfo info;
        info.setAgglomeration(p.mg_agglomeration);
        info.setConsolidation(p.mg_consolidation);
        info.setMaxCoarseningLevel(p.mg_max_coarsening_lev);

        op = std::make_unique<MLABecLaplacian>(Vector<Geometry>{gm},
                                               Vector<BoxArray>{ba},
                                               Vector<DistributionMapping>{dm},
                                               info);

        // Homogeneous Neumann matches the zero-gradient extrapolation in
        // fill_ghosts (an adiabatic wall); periodic directions are taken
        // from the Geometry so the two cannot disagree.
        Array<LinOpBCType,AMREX_SPACEDIM> lobc, hibc;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (gm.isPeriodic(idim)) {
                lobc[idim] = hibc[idim] = LinOpBCType::Periodic;
            } else {
                lobc[idim] = hibc[idim] = LinOpBCType::Neumann;
            }
        }
        op->setDomainBC(lobc, hibc);

        // nullptr = homogeneous. Both uses solve for an INCREMENT, whose
        // boundary condition is homogeneous even when the state's is not.
        op->setMaxOrder(2);
        op->setLevelBC(0, nullptr);
    }

    // ---------------------------------------------------------------
    // Rebuild the coefficients at the current Newton iterate. Called
    // once per Newton iteration: the coefficients depend only on the
    // iterate, so they are constant across a GMRES solve.
    void update (const MultiFab& state, const MultiFab& Eh,
                 Real dt, thd_diff::Workspace* ws)
    {
        dt_cur = dt;

        thd_diff::compute_coeffs(state, ws->coef);
        thd_diff::compute_face_coeffs(ws->coef, ws->fk, geom);

        if (!need_mg()) { build_diag(state, Eh, dt, ws); return; }

        // alpha = E_e/Te. Guarded the same way the Jacobi diagonal is:
        // an imperfect coefficient costs iterations, not accuracy, but a
        // non-finite one costs the whole solve.
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(alpha, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const s  = state.const_array(mfi);
            Array4<Real const> const cc = ws->coef.const_array(mfi);
            Array4<Real const> const eh = Eh.const_array(mfi);
            Array4<Real>       const a  = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const Real E    = amrex::max(s(i,j,k,UEINT), 1.0e-300_rt);
                const Real Ee   = amrex::max(E - eh(i,j,k), 1.0e-300_rt);
                const Real Te_c = amrex::max(cc(i,j,k,thd_diff::CTE), 1.0e-300_rt);
                Real av = Ee / Te_c;
                if (!amrex::Math::isfinite(av) || av <= 0.0_rt) { av = 1.0e-300_rt; }
                a(i,j,k) = av;
            });
        }

        // A = 1, B = dt  ->  operator = alpha*w - dt*div(kappa grad w).
        //
        // setBCoeffs COPIES into the operator's own storage, which
        // matters: ws->fk is scratch and compute_L overwrites it on
        // every subsequent residual evaluation.
        op->setScalars(1.0_rt, dt);
        op->setACoeffs(0, alpha);
        op->setBCoeffs(0, Array<MultiFab const*,AMREX_SPACEDIM>{
                              AMREX_D_DECL(&ws->fk[0], &ws->fk[1], &ws->fk[2])});

        // Rebuilt because MLMG caches per-coefficient state; the grid
        // hierarchy inside `op` is NOT rebuilt, which is the expensive
        // part.
        mlmg = std::make_unique<MLMG>(*op);
        mlmg->setVerbose(par.mg_verbose);
        mlmg->setBottomVerbose(par.mg_bottom_verbose);
        mlmg->setPreSmooth(par.mg_pre_smooth);
        mlmg->setPostSmooth(par.mg_post_smooth);
        if (par.mg_bottom_smoother) {
            mlmg->setBottomSolver(MLMG::BottomSolver::smoother);
        }
    }

    // ---------------------------------------------------------------
    // x <- (approximately) M^-1 x, in energy space, in place.
    //
    // A FIXED number of V-cycles. Not a tolerance: GMRES needs the
    // preconditioner to be a fixed linear operator, and a
    // tolerance-controlled inner solve makes M^-1 depend on its own
    // right-hand side.
    void apply_inplace (MultiFab& x)
    {
        if (!need_mg()) {
            MultiFab::Divide(x, diag, 0, 0, 1, 0);
            return;
        }

        MultiFab::Copy(rhs, x, 0, 0, 1, 0);
        wsol.setVal(0.0_rt);

        mlmg->setFixedIter(par.mg_fixed_iter);
        mlmg->solve(Vector<MultiFab*>{&wsol},
                    Vector<MultiFab const*>{&rhs},
                    0.0_rt, 0.0_rt);

        MultiFab::Copy(x, wsol, 0, 0, 1, 0);
        MultiFab::Multiply(x, alpha, 0, 0, 1, 0);   // dE = alpha*w
    }

    // ---------------------------------------------------------------
    // MODE_MG: solve M dE = -F to a tolerance and return the direction
    // directly. This is Newton with kappa' dropped, i.e. exactly the
    // Picard/frozen-coefficient linearization
    //
    //   alpha*Te^{k+1} - dt*div(kappa grad Te^{k+1})
    //       = E^n - E^k + alpha*Te^k,
    //
    // rearranged for the increment. No Krylov layer, no fd_eps.
    //
    // Returns the final MLMG residual norm.
    Real newton_direction (const MultiFab& F, MultiFab& dE)
    {
        AMREX_ALWAYS_ASSERT(need_mg());

        MultiFab::Copy(rhs, F, 0, 0, 1, 0);
        rhs.mult(-1.0_rt);
        wsol.setVal(0.0_rt);

        mlmg->setFixedIter(0);          // tolerance-controlled
        mlmg->setMaxIter(par.mg_max_iter);
        const Real res = mlmg->solve(Vector<MultiFab*>{&wsol},
                                     Vector<MultiFab const*>{&rhs},
                                     par.mg_rtol, par.mg_atol);

        MultiFab::Copy(dE, wsol, 0, 0, 1, 0);
        MultiFab::Multiply(dE, alpha, 0, 0, 1, 0);
        return res;
    }

private:

    // The exact diagonal of the discretized Jacobian,
    //
    //   d = 1 + dt * sum_dir m_dir * (k_lo + k_hi) * dTe/dE,
    //
    // using the same face coefficients and cell metric as the operator,
    // so the two cannot drift apart. dTe/dE = Te/E_e is exact for a
    // gamma-law electron gas. It never enters the residual: an imperfect
    // diagonal costs iterations, not accuracy.
    void build_diag (const MultiFab& state, const MultiFab& Eh,
                     Real dt, thd_diff::Workspace* ws)
    {
        // *Array() returns by value; geom.CellSize() would be a host
        // pointer captured into a device lambda.
        const amrex::GpuArray<Real,AMREX_SPACEDIM> problo = geom.ProbLoArray();
        const amrex::GpuArray<Real,AMREX_SPACEDIM> dx     = geom.CellSizeArray();
        const int coord = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(diag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const s  = state.const_array(mfi);
            Array4<Real const> const cc = ws->coef.const_array(mfi);
            Array4<Real const> const eh = Eh.const_array(mfi);
            AMREX_D_TERM(Array4<Real const> const fx = ws->fk[0].const_array(mfi);,
                         Array4<Real const> const fy = ws->fk[1].const_array(mfi);,
                         Array4<Real const> const fz = ws->fk[2].const_array(mfi);)
            Array4<Real> const d = diag.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const Real Te_c = cc(i,j,k,thd_diff::CTE);

                Real stiff = 0.0_rt;
                for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
                    Real kl = 0.0_rt, kr = 0.0_rt;
                    if (idir == 0) { kl = fx(i,j,k); kr = fx(i+1,j,k); }
#if AMREX_SPACEDIM >= 2
                    if (idir == 1) { kl = fy(i,j,k); kr = fy(i,j+1,k); }
#endif
#if AMREX_SPACEDIM == 3
                    if (idir == 2) { kl = fz(i,j,k); kr = fz(i,j,k+1); }
#endif
                    stiff += thd_diff::cell_metric(i, j, k, idir,
                                                   problo.data(), dx.data(), coord)
                             * (kl + kr);
                }

                const Real E    = amrex::max(s(i,j,k,UEINT), 1.0e-300_rt);
                const Real Ee   = amrex::max(E - eh(i,j,k), 1.0e-300_rt);
                const Real dTdE = Te_c / Ee;

                Real dg = 1.0_rt + dt * stiff * dTdE;
                if (!amrex::Math::isfinite(dg) || dg < 1.0_rt) { dg = 1.0_rt; }
                d(i,j,k) = dg;
            });
        }
    }
};


// Left-preconditioned GMRES for J dE = -F.
//
// The matvec stays matrix-free, so the FULL Jacobian -- including the
// non-symmetric dkappa/dTe term that cannot be expressed as an
// MLABecLaplacian -- is applied. Only the preconditioner is
// approximate.
//
// Modified Gram-Schmidt, with a second pass when orthogonalization
// collapses the norm (deep Krylov spaces lose orthogonality, and the
// Givens recurrence would still report convergence).
//
// The Hessenberg matrix is one flat vector, not a vector of rows: the
// row-of-vectors form allocated m+1 separate blocks per call.
inline int
gmres (const MultiFab& state, const MultiFab& F0, const MultiFab& En,
       const MultiFab& Eh, Precond& pc, MultiFab& dE,
       const Geometry& geom, Real dt, const Params& p,
       thd_diff::Workspace* ws, MultiFab& pert, MultiFab& base,
       MultiFab& w, Real* final_resid = nullptr)
{
    const BoxArray&            ba = F0.boxArray();
    const DistributionMapping& dm = F0.DistributionMap();

    dE.setVal(0.0_rt);

    // pert tracks `state` except for UEINT/UFX, which apply_update
    // rewrites each iteration; the full copy happens once here.
    MultiFab::Copy(pert, state, 0, 0, state.nComp(), state.nGrow());
    MultiFab::Copy(base, state, UEINT, 0, 1, 0);

    // r0 = -F0, preconditioned in place.
    MultiFab z(ba, dm, 1, 0);
    MultiFab::Copy(z, F0, 0, 0, 1, 0);
    z.mult(-1.0_rt);
    pc.apply_inplace(z);

    const Real beta = z.norm2(0);
    if (final_resid != nullptr) { *final_resid = 0.0_rt; }
    if (!(beta > 1.0e-300_rt)) { return 0; }

    int m = p.max_gmres;
    if (m <= 0) {
        // With PC_MG the count should be single digits and
        // dx-independent, so the size is set by the ceiling alone; with
        // PC_JACOBI it grows like dx^(-1/2) and the old grid-based sizing
        // is kept.
        if (pc.need_mg()) {
            m = amrex::min(40, p.gmres_ceiling);
        } else {
            const Box& dom = geom.Domain();
            int nmax = dom.length(0);
            for (int d = 1; d < AMREX_SPACEDIM; ++d) { nmax = amrex::max(nmax, dom.length(d)); }
            m = amrex::max(amrex::min(2 * nmax, p.gmres_ceiling), 40);
        }
    }

    std::vector<std::unique_ptr<MultiFab>> V;
    V.reserve(m + 1);
    V.push_back(std::make_unique<MultiFab>(ba, dm, 1, 0));
    MultiFab::Copy(*V[0], z, 0, 0, 1, 0);
    V[0]->mult(1.0_rt / beta);

    // Flat Hessenberg, row stride m.
    std::vector<Real> H(static_cast<std::size_t>(m + 1) * m, 0.0_rt);
    auto Hat = [&H, m](int row, int col) -> Real& {
        return H[static_cast<std::size_t>(row) * m + col];
    };
    std::vector<Real> g(m + 1, 0.0_rt), cs(m, 0.0_rt), sn(m, 0.0_rt);
    g[0] = beta;

    int kend = 0;
    for (int k = 0; k < m; ++k) {
        kend = k + 1;

        // Matrix-free J*v ~= [F(E + eps*v) - F(E)] / eps, then
        // preconditioned in place. Both norms must be the SAME norm;
        // mixing L2 for the state with L-infinity for the direction
        // inflates eps by ~sqrt(N).
        {
            const Real norm_E = state.norm2(UEINT);
            const Real norm_v = V[k]->norm2(0);
            const Real eps = p.fd_eps * (norm_E + 1.0_rt) / (norm_v + 1.0e-300_rt);

            apply_update(pert, base, *V[k], eps, Eh, geom);
            residual(pert, En, w, geom, dt, ws);   // w = F(E + eps*v)
            MultiFab::Saxpy(w, -1.0_rt, F0, 0, 0, 1, 0);
            w.mult(1.0_rt / eps);
        }
        pc.apply_inplace(w);

        const Real wnorm_before = w.norm2(0);

        for (int i = 0; i <= k; ++i) {
            Hat(i,k) = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
            MultiFab::Saxpy(w, -Hat(i,k), *V[i], 0, 0, 1, 0);
        }
        Hat(k+1,k) = w.norm2(0);

        if (p.reorth_tol > 0.0_rt && Hat(k+1,k) < p.reorth_tol * wnorm_before) {
            for (int i = 0; i <= k; ++i) {
                const Real corr = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
                Hat(i,k) += corr;
                MultiFab::Saxpy(w, -corr, *V[i], 0, 0, 1, 0);
            }
            Hat(k+1,k) = w.norm2(0);
        }

        const bool breakdown = !(Hat(k+1,k) > 1.0e-300_rt);

        if (!breakdown && k + 1 < m) {
            V.push_back(std::make_unique<MultiFab>(ba, dm, 1, 0));
            MultiFab::Copy(*V[k+1], w, 0, 0, 1, 0);
            V[k+1]->mult(1.0_rt / Hat(k+1,k));
        }

        for (int i = 0; i < k; ++i) {
            const Real t = cs[i]*Hat(i,k) + sn[i]*Hat(i+1,k);
            Hat(i+1,k)   = -sn[i]*Hat(i,k) + cs[i]*Hat(i+1,k);
            Hat(i,k)     = t;
        }
        Real den = std::sqrt(Hat(k,k)*Hat(k,k) + Hat(k+1,k)*Hat(k+1,k));
        if (den < 1.0e-300_rt) { den = 1.0e-300_rt; }
        cs[k] = Hat(k,k)   / den;
        sn[k] = Hat(k+1,k) / den;
        Hat(k,k)   = cs[k]*Hat(k,k) + sn[k]*Hat(k+1,k);
        Hat(k+1,k) = 0.0_rt;

        const Real gt = cs[k]*g[k];
        g[k+1] = -sn[k]*g[k];
        g[k]   = gt;

        if (final_resid != nullptr) { *final_resid = std::abs(g[k+1]) / beta; }

        if (breakdown) { break; }
        if (static_cast<std::size_t>(k + 1) >= V.size()) { break; }
        if (std::abs(g[k+1]) / beta < p.gmres_rtol) { break; }
    }

    std::vector<Real> y(kend, 0.0_rt);
    for (int i = kend - 1; i >= 0; --i) {
        Real s = g[i];
        for (int j = i + 1; j < kend; ++j) { s -= Hat(i,j) * y[j]; }
        y[i] = (std::abs(Hat(i,i)) > 1.0e-300_rt) ? s / Hat(i,i) : 0.0_rt;
    }
    for (int i = 0; i < kend; ++i) {
        MultiFab::Saxpy(dE, y[i], *V[i], 0, 0, 1, 0);
    }

    return kend;
}


// Result cache. Castro evaluates ext_src more than once per step from
// the same old-time state; the key is (time, dt), the BoxArray and a
// three-number checksum of the input energy.

struct SolveCache {
    bool     valid = false;
    Real     time  = -1.0_rt;
    Real     dt    = -1.0_rt;
    Real     sum   = 0.0_rt;
    Real     lo    = 0.0_rt;
    Real     hi    = 0.0_rt;
    BoxArray ba;
    MultiFab dsrc_e;
    MultiFab dsrc_fx;
    bool     converged = false;
};

inline SolveCache& solve_cache ()
{
    static SolveCache c;
    return c;
}


// Top-level driver.
//
//   state_in : full state, const, with >= 1 FillPatch'd ghost cell.
//   dsrc_e   : OUT, (E^{n+1} - E^n)/dt        -> caller adds to UEINT, UEDEN.
//   dsrc_fx  : OUT, (rho f^{n+1} - rho f^n)/dt -> caller adds to UFX.
//
// The aux source is the EXACT change implied by freezing E_h, valid for
// any dE. The line search enforces E > E_h, so f stays in [0,1] by
// construction. problem_source must NOT also put the conduction energy
// through -(f/e)*Edot, or the aux change is counted twice.
//
// Returns true if Newton converged.

inline bool
solve (const MultiFab& state_in, MultiFab& dsrc_e, MultiFab& dsrc_fx,
       const Geometry& geom, Real dt, int nGrow, const Params& p = Params(),
       Real time = -1.0_rt)
{
    const BoxArray&            ba = state_in.boxArray();
    const DistributionMapping& dm = state_in.DistributionMap();
    const int ncomp = state_in.nComp();

    // ---- MODE_OFF ---------------------------------------------------
    if (p.mode == MODE_OFF) {
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        return true;
    }

    const bool use_krylov = (p.mode == MODE_IMPLICIT);

    // ---- cache lookup -----------------------------------------------
    Real ck_sum = 0.0_rt, ck_lo = 0.0_rt, ck_hi = 0.0_rt;
    if (p.use_cache && time >= 0.0_rt) {
        ck_sum = state_in.norm2(UEINT);
        ck_lo  = state_in.min(UEINT, 0);
        ck_hi  = state_in.max(UEINT, 0);

        SolveCache& c = solve_cache();
        if (c.valid && c.time == time && c.dt == dt &&
            c.sum == ck_sum && c.lo == ck_lo && c.hi == ck_hi &&
            c.ba == state_in.boxArray())
        {
            MultiFab::Copy(dsrc_e,  c.dsrc_e,  0, 0, 1, 0);
            MultiFab::Copy(dsrc_fx, c.dsrc_fx, 0, 0, 1, 0);
            if (p.verbose) {
                amrex::Print() << "[THD-JFNK] repeat solve, reusing result\n";
            }
            return c.converged;
        }
    }

    // Working copy INCLUDING the caller's ghosts. Under AMR only
    // FillPatch (done by Castro's expand_state) can fill coarse-fine
    // ghosts; copying the valid region alone discarded them and the
    // stencil then read stale memory along every refinement interface.
    // Keeping them freezes the coarse-fine ghosts at their interpolated
    // values for the solve -- a Dirichlet condition from the coarse
    // level, lagged by a step but stable.
    //
    // NOTE: the MLMG operator below is single-level and takes
    // HOMOGENEOUS conditions at the coarse-fine interface. As a
    // preconditioner (MODE_IMPLICIT) that is only an iteration-count
    // question, since the matrix-free residual still sees the real
    // ghosts. In MODE_MG it also lags the interface flux, the same way
    // the existing scheme does. Proper flux matching would need
    // MLLinOp::setCoarseFineBC (or a composite multi-level solve), which
    // is the natural next step here.
    const int nG = amrex::max(nGrow, 1);
    const int ncopy = amrex::min(nG, state_in.nGrow());
    MultiFab state(ba, dm, ncomp, nG);
    MultiFab::Copy(state, state_in, 0, 0, ncomp, ncopy);
    fill_ghosts(state, geom);

    if (!(state.min(URHO, nG) > 0.0_rt) ||
        !amrex::Math::isfinite(state.min(UEINT, nG)) ||
        !amrex::Math::isfinite(state.max(UEINT, nG)))
    {
        amrex::Print() << "[THD-JFNK] WARNING: ghost cells look invalid (rho min = "
                       << state.min(URHO, nG) << "). Caller must supply a "
                       << "FillPatch'd state with >= 1 ghost cell.\n";
    }

    // E_h = E*f, frozen for the whole sub-step: electron conduction does
    // not move energy into or out of the heavies.
    MultiFab Eh(ba, dm, 1, nG);
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(Eh, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real>       const eh = Eh.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real rho = s(i,j,k,URHO);
            const Real E   = s(i,j,k,UEINT);
            if (rho > 0.0_rt && amrex::Math::isfinite(E)) {
                const Real f = amrex::min(amrex::max(s(i,j,k,UFX)/rho, 0.0_rt), 1.0_rt);
                eh(i,j,k) = E * f;
            } else {
                eh(i,j,k) = 0.0_rt;
            }
        });
    }

    MultiFab En(ba, dm, 1, 0);
    MultiFab::Copy(En, state, UEINT, 0, 1, 0);

    // Everything below is allocated ONCE per solve. Allocating inside
    // the Newton or Krylov loops cost hundreds of MultiFab constructions
    // per solve.
    MultiFab F(ba, dm, 1, 0), dE(ba, dm, 1, 0);
    MultiFab base(ba, dm, 1, 0);
    MultiFab trial(ba, dm, ncomp, nG);              // line-search trial
    MultiFab Ftrial(ba, dm, 1, 0);

    // Only the Krylov path needs a perturbed state and a work vector.
    MultiFab pert, w;
    if (use_krylov) {
        pert.define(ba, dm, ncomp, nG);
        w   .define(ba, dm, 1, 0);
    }

    thd_diff::Workspace wsp;
    wsp.define(ba, dm);

    // MODE_MG needs the MG machinery whatever `precond` says; MODE_IMPLICIT
    // needs it only when PC_MG is selected.
    Precond pc;
    pc.type = (p.mode == MODE_MG) ? PC_MG : p.precond;
    pc.define(ba, dm, geom, p, pc.type == PC_MG);

    // Te bounds before the solve, for the maximum-principle check below.
    Real Te_lo_old = 0.0_rt, Te_hi_old = 0.0_rt;
    if (p.verbose) {
        thd_diff::compute_coeffs(state, wsp.coef);
        Te_lo_old = wsp.coef.min(thd_diff::CTE);
        Te_hi_old = wsp.coef.max(thd_diff::CTE);
    }

    residual(state, En, F, geom, dt, &wsp);
    Real Fnorm = F.norm2(0);

    if (!amrex::Math::isfinite(Fnorm)) {
        amrex::Print() << "[THD-JFNK] ERROR: initial residual not finite.\n"
                       << "          rho  valid/ghost min = " << state.min(URHO,0)
                       << " / " << state.min(URHO,nG) << "\n"
                       << "          eint valid/ghost min = " << state.min(UEINT,0)
                       << " / " << state.min(UEINT,nG) << "\n"
                       << "          Skipping diffusion this step.\n";
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        return false;
    }

    const Real escale = amrex::max(En.norm2(0), 1.0e-300_rt);
    const Real tol    = p.newton_rtol * escale;
    bool converged    = (Fnorm <= tol);

    if (p.verbose) {
        // |F(E^n)| = |dt*L(E^n)|, so this ratio is the explicit stability
        // parameter. Above ~0.5 an explicit treatment would be unstable.
        const Real stab = Fnorm / escale;
        amrex::Print() << "[THD-JFNK] enter  |F|=" << Fnorm << "  tol=" << tol
                       << "  |dt*L|/|E|=" << stab
                       << (stab > 0.5_rt ? "  (explicit would be unstable)" : "")
                       << "\n";
    }

    for (int it = 0; it < p.max_newton && !converged; ++it) {

        pc.update(state, Eh, dt, &wsp);

        int  nkr  = 0;
        Real kres = 0.0_rt;

        if (use_krylov) {
            nkr = gmres(state, F, En, Eh, pc, dE, geom, dt, p,
                        &wsp, pert, base, w, &kres);

            // A silent inner failure returns a partial vector and Newton
            // then follows a wrong direction with no indication.
            if (kres > p.gmres_rtol) {
                amrex::Print() << "[THD-JFNK] WARNING: GMRES stopped at " << nkr
                               << " iterations, relative residual " << kres
                               << " > " << p.gmres_rtol
                               << ". Newton direction is inexact.\n";
            }
        } else {
            // MODE_MG: the MLMG solve IS the linear solve.
            kres = pc.newton_direction(F, dE);
        }

        MultiFab::Copy(base, state, UEINT, 0, 1, 0);

        Real lambda = 1.0_rt;
        bool ok = false;
        Real Ftrial_norm = Fnorm;

        for (int ls = 0; ls < p.max_linesearch && !ok; ++ls) {
            MultiFab::Copy(trial, state, 0, 0, ncomp, nG);
            apply_update(trial, base, dE, lambda, Eh, geom);

            // Electron conduction can only move electron energy, so
            // E > E_h. Enforcing it here bounds |dE| by the available
            // electron energy and keeps f in [0,1], with no artificial cap.
            if (!(min_electron_energy(trial, Eh) > 0.0_rt)) {
                lambda *= 0.5_rt;
                continue;
            }

            residual(trial, En, Ftrial, geom, dt, &wsp);
            Ftrial_norm = Ftrial.norm2(0);

            if (Ftrial_norm < Fnorm || Ftrial_norm <= tol) { ok = true; }
            else                                           { lambda *= 0.5_rt; }
        }

        if (!ok) {
            // Committing a step known to increase the residual is how a
            // converging solve turns into a diverging one.
            //
            // MODE_MG hits this more readily than MODE_IMPLICIT: with
            // kappa' dropped, the direction is only an approximate
            // Newton direction, and near a steep front it can stop being
            // a descent direction before the residual is at tolerance.
            // If this fires often, switch to mode = implicit.
            if (p.verbose) {
                amrex::Print() << "[THD-JFNK]   it=" << it
                               << "  line search failed, keeping previous iterate\n";
            }
            break;
        }

        MultiFab::Copy(state, trial, 0, 0, ncomp, nG);
        MultiFab::Copy(F, Ftrial, 0, 0, 1, 0);
        Fnorm = Ftrial_norm;
        converged = (Fnorm <= tol);

        if (p.verbose) {
            amrex::Print() << "[THD-JFNK]   it=" << it;
            if (use_krylov) {
                amrex::Print() << "  krylov=" << nkr << "  kres=" << kres;
            } else {
                amrex::Print() << "  mlmg_res=" << kres;
            }
            amrex::Print() << "  lambda=" << lambda << "  |F|=" << Fnorm
                           << (converged ? "  CONVERGED" : "") << "\n";
        }
    }

    // ---- source back-out --------------------------------------------
    //
    // ENERGY: bounded by ONE GLOBAL factor. The update is a flux
    // divergence, so what one cell loses its neighbours gain and the sum
    // telescopes; clamping a single cell would create energy. Scaling the
    // whole field by a constant keeps it conservative. Only cooling is
    // bounded -- the electron energy is a real budget for it -- while
    // heating needs no cap, since the exact aux relation handles any dE.
    //
    // AUX: bounded PER CELL. UFX carries no energy, only the
    // electron/heavy split, so a per-cell bound cannot break the
    // telescoping. It is needed because a cold cell at the front moves f
    // across nearly the whole [0,1] range in one application, and Castro
    // applies the source to a state hydro has advanced.
    constexpr Real max_frac_drop = 0.9_rt;
    constexpr Real aux_safety    = 0.4_rt;   // < 0.5: headroom for old/new combination
    const Real Ee_floor = 1.0e-10_rt * amrex::max(En.max(0), 1.0e-300_rt);

    MultiFab& dEmf  = wsp.lap;       // reuse: L is no longer needed
    MultiFab& dFmf  = wsp.scratch;
    MultiFab  ratio(ba, dm, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dEmf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const snew = state.const_array(mfi);
        Array4<Real const> const sold = state_in.const_array(mfi);
        Array4<Real const> const eh   = Eh.const_array(mfi);
        Array4<Real>       const de   = dEmf.array(mfi);
        Array4<Real>       const df   = dFmf.array(mfi);
        Array4<Real>       const rt   = ratio.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real Eold = sold(i,j,k,UEINT);
            const Real Ee   = Eold - eh(i,j,k);

            Real dE_c = snew(i,j,k,UEINT) - Eold;
            if (!amrex::Math::isfinite(dE_c)) { dE_c = 0.0_rt; }
            de(i,j,k) = dE_c;

            Real dF = snew(i,j,k,UFX) - sold(i,j,k,UFX);
            if (!amrex::Math::isfinite(dF)) { dF = 0.0_rt; }

            const Real rho_c = sold(i,j,k,URHO);
            const Real ufx_c = sold(i,j,k,UFX);
            if (rho_c > 0.0_rt) {
                const Real lo = -aux_safety * amrex::max(ufx_c, 0.0_rt);
                const Real hi =  aux_safety * amrex::max(rho_c - ufx_c, 0.0_rt);
                dF = amrex::min(amrex::max(dF, lo), hi);
            } else {
                dF = 0.0_rt;
            }
            df(i,j,k) = dF;

            Real r = 1.0_rt;
            if (dE_c < 0.0_rt && Ee > Ee_floor) {
                r = amrex::min(1.0_rt, max_frac_drop * Ee / (-dE_c));
            }
            rt(i,j,k) = r;
        });
    }

    Real lam = ratio.min(0);
    if (!amrex::Math::isfinite(lam) || lam < 0.0_rt) { lam = 0.0_rt; }
    lam = amrex::min(lam, 1.0_rt);

    MultiFab::Copy(dsrc_e,  dEmf, 0, 0, 1, 0);
    MultiFab::Copy(dsrc_fx, dFmf, 0, 0, 1, 0);
    dsrc_e .mult(p.source_scale * lam / dt);
    dsrc_fx.mult(p.source_scale * lam / dt);

    // ---- diagnostics -------------------------------------------------
    if (p.verbose) {
        amrex::Print() << "[THD-JFNK] lambda=" << lam
                       << (lam < 0.99_rt ? "  (source damped)" : "") << "\n";

        // Maximum principle applies to Te, the quantity actually
        // diffused -- NOT to E, whose bounds legitimately widen when n_e
        // varies in space.
        thd_diff::compute_coeffs(state, wsp.coef);
        const Real Te_lo_new = wsp.coef.min(thd_diff::CTE);
        const Real Te_hi_new = wsp.coef.max(thd_diff::CTE);
        const Real slack = 1.0e-8_rt * amrex::max(std::abs(Te_hi_old), 1.0_rt);
        const bool violated = (Te_hi_new > Te_hi_old + slack) ||
                              (Te_lo_new < Te_lo_old - slack);
        amrex::Print() << "[THD-JFNK] Te [" << Te_lo_old << ", " << Te_hi_old
                       << "] -> [" << Te_lo_new << ", " << Te_hi_new << "]"
                       << (violated ? "  *** MAX-PRINCIPLE VIOLATED ***" : "  ok")
                       << "\n"
                       << "[THD-JFNK] dsrc_e  " << dsrc_e.min(0)  << " / " << dsrc_e.max(0)
                       << "   dsrc_fx " << dsrc_fx.min(0) << " / " << dsrc_fx.max(0) << "\n";
    }

    if (p.use_cache && time >= 0.0_rt) {
        SolveCache& c = solve_cache();
        c.dsrc_e .define(ba, dm, 1, 0);
        c.dsrc_fx.define(ba, dm, 1, 0);
        MultiFab::Copy(c.dsrc_e,  dsrc_e,  0, 0, 1, 0);
        MultiFab::Copy(c.dsrc_fx, dsrc_fx, 0, 0, 1, 0);
        c.time = time;   c.dt = dt;
        c.sum  = ck_sum; c.lo = ck_lo; c.hi = ck_hi;
        c.ba   = state_in.boxArray();
        c.converged = converged;
        c.valid = true;
    }

    return converged;
}

} // namespace thd_jfnk

#endif