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
// THE LINEAR SYSTEM
// -----------------
// The Frechet derivative of F, with s = dTe/dE = Te/E_e (exact for a
// gamma-law electron gas), is
//
//   J v = v - dt*div( kappa grad(s v) ) - dt*div( v s kappa' grad Te ).
//
// It is never assembled. The matvec is a finite difference of the real
// residual, so the FULL Jacobian is applied -- including the
// non-symmetric kappa' term (kappa' = 2.5*kappa/Te for Spitzer, so
// comparable to or larger than the second term at a steep front).
//
// The preconditioner is the exact diagonal of the discretized J,
//
//   d = 1 + dt * sum_dir m_dir*(k_lo + k_hi) * dTe/dE,
//
// built from the same face coefficients and cell metric the operator
// uses, so the two cannot drift apart. Applying it is one Divide.
//
// This is cheap but it does not fix the conditioning: cond(J) ~
// 1 + dt*chi/dx^2, and a diagonal scaling leaves that alone, so the
// Krylov iteration count grows as the grid is refined (roughly
// dx^(-1/2) in practice for this operator). The Krylov budget below is
// therefore sized from the domain rather than left at a fixed default.
// If that budget starts binding at high resolution, the fix is a
// preconditioner that sees the off-diagonal coupling -- a shifted
// MLABecLaplacian with alpha = E_e/Te, beta = kappa, solved with a
// fixed number of MLMG V-cycles -- not a larger budget.
//
//
// THE KRYLOV LAYER IS amrex::GMRES
// --------------------------------
// The Arnoldi/Givens machinery is amrex::GMRES<MultiFab, JacobianOp>
// (AMReX_GMRES.H, 25.06), not a local reimplementation. This file
// supplies only the operator concept it asks for: makeVecRHS/makeVecLHS,
// apply (the finite-difference J*v), precond (the Divide), and the
// vector primitives norm2/dotProduct/scale/setToZero/assign/increment/
// linComb, all of which act on the valid region only so that RHS-shaped
// (no ghost) and LHS-shaped (one ghost) vectors can be mixed the way
// GMRES mixes them.
//
// Two properties of the library version worth knowing:
//
//   * It is RIGHT-preconditioned (cycle() calls precond then apply, and
//     build_solution applies precond once to the assembled Krylov
//     combination). So the residual it monitors is the TRUE residual
//     |b - J dE|, and gmres_rtol is a tolerance on the actual linear
//     residual. Applying P to a sum is only the same as summing its
//     applications if P is a fixed linear operator; a diagonal is, so
//     nothing here needs FGMRES.
//
//   * Orthogonalization is two unconditional unmodified Gram-Schmidt
//     passes, so there is no reorthogonalization threshold to tune, and
//     it RESTARTS at gmres_restart rather than growing the basis without
//     bound. Restarting a stiff diffusion solve can stagnate, so the
//     default sizing below gives one long cycle (max_gmres = restart
//     length) -- the old behaviour. Set max_gmres above gmres_restart to
//     actually allow restarts.
//
#include <thd_diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_GMRES.H>
#include <string>

using namespace amrex;

namespace thd_jfnk {

//   MODE_OFF       contribute nothing. Use with castro.diffuse_temp = 1
//                  to run Castro's own explicit diffusion instead.
//   MODE_IMPLICIT  backward-Euler JFNK, exact Jacobian, amrex::GMRES,
//                  diagonal preconditioner.

enum DiffMode { MODE_OFF = 0, MODE_IMPLICIT = 1 };

struct Params {
    DiffMode mode = MODE_IMPLICIT;

    // Castro evaluates ext_src more than once per step from the same
    // old-time state. Reuse the result when the input is identical;
    // guarded by a checksum, so a mismatch falls through to a real solve.
    bool use_cache = true;

    // Multiplies the returned source. Castro combines the old-time and
    // new-time evaluations; try 0.5 to test whether that over-applies.
    Real source_scale = 1.0_rt;

    // Newton. Tolerance is relative to the energy scale, not to the
    // initial residual, which would be self-referential.
    int  max_newton     = 20;
    Real newton_rtol    = 1.0e-6_rt;
    int  max_linesearch = 8;
    Real fd_eps         = 1.0e-7_rt;   // finite-difference step for J*v

    // Krylov. gmres_rtol is relative to |F|, the true linear residual,
    // because amrex::GMRES is right-preconditioned.
    //
    // gmres_restart = 0 auto-sizes the Krylov basis from the grid --
    // 2*nmax, floored at 40 and capped at gmres_ceiling -- because with
    // a diagonal preconditioner the iteration count is dx-dependent.
    // max_gmres = 0 then matches it, i.e. a single un-restarted cycle.
    int  gmres_restart  = 0;
    int  max_gmres      = 0;
    int  gmres_ceiling  = 120;
    Real gmres_rtol     = 1.0e-4_rt;
    int  gmres_verbose  = 0;           // 2 prints per-iteration residuals

    // Per-step diagnostics. Costs two extra coefficient evaluations.
    bool verbose = false;
};


// Runtime parameters, read once from the inputs file.
//   JFNK               thd_diffusion.mode  = implicit
//                      castro.diffuse_temp = 0
//   explicit baseline  thd_diffusion.mode  = off
//                      castro.diffuse_temp = 1

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
        else if (m == "mg") {
            amrex::Abort("thd_diffusion.mode = 'mg' has been removed. The only "
                         "modes are 'off' and 'implicit'.");
        }
        else { amrex::Abort("thd_diffusion.mode must be 'off' or 'implicit'"); }
    }

    // The MLMG preconditioner has been removed; catch stale inputs files
    // rather than silently running with a different preconditioner than
    // the one they ask for.
    std::string pcs;
    if (pp.query("precond", pcs) && pcs != "jacobi") {
        amrex::Abort("thd_diffusion.precond = '" + pcs + "' is no longer "
                     "available: the diagonal is the only preconditioner. "
                     "Remove the parameter or set it to 'jacobi'.");
    }

    pp.query("source_scale",   p.source_scale);
    pp.query("max_newton",     p.max_newton);
    pp.query("newton_rtol",    p.newton_rtol);
    pp.query("max_linesearch", p.max_linesearch);
    pp.query("gmres_restart",  p.gmres_restart);
    pp.query("max_gmres",      p.max_gmres);
    pp.query("gmres_ceiling",  p.gmres_ceiling);
    pp.query("gmres_rtol",     p.gmres_rtol);
    pp.query("gmres_verbose",  p.gmres_verbose);
    pp.query("fd_eps",         p.fd_eps);

    int itmp = p.verbose ? 1 : 0;
    if (pp.query("verbose", itmp))   { p.verbose = itmp; }
    itmp = p.use_cache ? 1 : 0;
    if (pp.query("use_cache", itmp)) { p.use_cache = itmp; }

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
        amrex::Print() << "[THD-JFNK] mode = "
                       << ((p.mode == MODE_OFF) ? "off" : "implicit")
                       << ", castro.diffuse_temp = " << castro_diffuse_temp << "\n";
    }
    return p;
}


// Ghost fill.
//
// FillBoundary covers same-level and periodic ghosts. Ghosts OUTSIDE the
// domain are then filled by zero-gradient extrapolation, a no-flux
// (adiabatic) thermal wall.
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
// The diagonal preconditioner.
//
//   d = 1 + dt * sum_dir m_dir * (k_lo + k_hi) * dTe/dE
//
// is the exact diagonal of the discretized Jacobian, in energy space,
// using the same face coefficients and cell metric as the operator so
// the two cannot drift apart. dTe/dE = Te/E_e is exact for a gamma-law
// electron gas; the factor is what converts the temperature-space
// stencil into the energy-space one Newton actually iterates on.
//
// It never enters the residual: an imperfect diagonal costs iterations,
// not accuracy.
//
// dt -> 0 sanity check: d -> 1, so P^-1 -> I, matching J -> I.
// ====================================================================

struct Precond {

    MultiFab diag;
    Geometry geom;

    // Allocated once per solve.
    void define (const BoxArray& ba, const DistributionMapping& dm,
                 const Geometry& gm)
    {
        geom = gm;
        diag.define(ba, dm, 1, 0);
    }

    // Rebuild at the current Newton iterate. Called once per Newton
    // iteration: the coefficients depend only on the iterate, so the
    // preconditioner is constant across a GMRES solve -- which is what
    // right-preconditioned GMRES requires.
    void update (const MultiFab& state, const MultiFab& Eh,
                 Real dt, thd_diff::Workspace* ws)
    {
        thd_diff::compute_coeffs(state, ws->coef);
        thd_diff::compute_face_coeffs(ws->coef, ws->fk, geom);

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

    // lhs = P^-1 r. Valid region only, so lhs may be an LHS-shaped
    // (ghosted) vector and r an RHS-shaped one, which is how
    // amrex::GMRES calls this.
    void apply (MultiFab& lhs, const MultiFab& r)
    {
        MultiFab::Copy(lhs, r, 0, 0, 1, 0);
        MultiFab::Divide(lhs, diag, 0, 0, 1, 0);
    }
};


// ====================================================================
// The operator concept amrex::GMRES<V,M> requires, with V = MultiFab.
//
// apply() is the only interesting member: the matrix-free J*v. The
// FULL Jacobian -- including the non-symmetric dkappa/dTe term -- is
// applied, because it is a finite difference of the real residual. Only
// the preconditioner is approximate.
//
// Everything else is a thin wrapper over MultiFab arithmetic, all of it
// restricted to the valid region (nghost = 0) so that the RHS-shaped
// vectors GMRES builds its Krylov basis from and the LHS-shaped vector
// it hands to precond()/apply() can be combined freely.
//
// The object holds only non-owning pointers; prepare() re-points it at
// the current Newton iterate. Nothing here allocates.
// ====================================================================

struct JacobianOp {

    using RT = amrex::Real;

    BoxArray            ba;
    DistributionMapping dm;
    int                 ncomp = 0;
    int                 nghost_state = 1;

    const MultiFab* state = nullptr;   // current Newton iterate
    const MultiFab* F0    = nullptr;   // F(state)
    const MultiFab* En    = nullptr;
    const MultiFab* Eh    = nullptr;
    const Geometry* geom  = nullptr;
    Precond*        pc    = nullptr;
    thd_diff::Workspace* ws = nullptr;

    MultiFab* pert = nullptr;          // scratch: perturbed state
    MultiFab* base = nullptr;          // scratch: UEINT of the iterate

    Real dt     = 0.0_rt;
    Real fd_eps = 1.0e-7_rt;
    Real norm_E = 0.0_rt;              // |UEINT| of the iterate, cached

    // Called once per Newton iteration. `pert` tracks `state` except for
    // UEINT/UFX, which apply_update rewrites on every matvec; the full
    // copy happens here rather than per matvec.
    void prepare (const MultiFab& a_state, const MultiFab& a_F0)
    {
        state = &a_state;
        F0    = &a_F0;
        MultiFab::Copy(*pert, a_state, 0, 0, ncomp, nghost_state);
        MultiFab::Copy(*base, a_state, UEINT, 0, 1, 0);
        norm_E = a_state.norm2(UEINT);
    }

    // ---- vector factories ------------------------------------------
    [[nodiscard]] MultiFab makeVecRHS () const { return MultiFab(ba, dm, 1, 0); }

    [[nodiscard]] MultiFab makeVecLHS () const
    {
        MultiFab mf(ba, dm, 1, 1);
        mf.setVal(0.0_rt);
        return mf;
    }

    // ---- vector primitives -----------------------------------------
    static RT norm2 (const MultiFab& v) { return v.norm2(0); }

    static RT dotProduct (const MultiFab& v1, const MultiFab& v2)
    {
        return MultiFab::Dot(v1, 0, v2, 0, 1, 0);
    }

    static void scale (MultiFab& v, RT fac) { v.mult(fac, 0, 1, 0); }

    static void setToZero (MultiFab& v) { v.setVal(0.0_rt); }

    static void assign (MultiFab& lhs, const MultiFab& rhs)
    {
        MultiFab::Copy(lhs, rhs, 0, 0, 1, 0);
    }

    static void increment (MultiFab& lhs, const MultiFab& rhs, RT a)
    {
        MultiFab::Saxpy(lhs, a, rhs, 0, 0, 1, 0);
    }

    static void linComb (MultiFab& lhs, RT a, const MultiFab& rhs_a,
                         RT b, const MultiFab& rhs_b)
    {
        MultiFab::LinComb(lhs, a, rhs_a, 0, b, rhs_b, 0, 0, 1, 0);
    }

    // ---- lhs = J*v --------------------------------------------------
    //
    // J*v ~= [F(E + eps*v) - F(E)] / eps.
    //
    // Both norms must be the SAME norm; mixing L2 for the state with
    // L-infinity for the direction inflates eps by ~sqrt(N). Under right
    // preconditioning v is already P^-1 of a unit Krylov vector, so its
    // scale is not O(1) and the normalisation is doing real work.
    void apply (MultiFab& lhs, const MultiFab& v)
    {
        const Real norm_v = norm2(v);
        if (!(norm_v > 0.0_rt)) { lhs.setVal(0.0_rt); return; }

        const Real eps = fd_eps * (norm_E + 1.0_rt) / norm_v;

        apply_update(*pert, *base, v, eps, *Eh, *geom);
        residual(*pert, *En, lhs, *geom, dt, ws);   // lhs = F(E + eps*v)
        MultiFab::Saxpy(lhs, -1.0_rt, *F0, 0, 0, 1, 0);
        lhs.mult(1.0_rt / eps, 0, 1, 0);
    }

    // ---- lhs = P^-1 rhs ---------------------------------------------
    void precond (MultiFab& lhs, const MultiFab& r) { pc->apply(lhs, r); }
};


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
    // per solve. (amrex::GMRES still builds and releases its own Krylov
    // basis per solve() call, i.e. once per Newton iteration.)
    MultiFab F(ba, dm, 1, 0);
    MultiFab mrhs(ba, dm, 1, 0);                    // -F, the GMRES rhs
    MultiFab dE(ba, dm, 1, 1);                      // GMRES solution vector
    MultiFab base(ba, dm, 1, 0);
    MultiFab pert(ba, dm, ncomp, nG);               // FD-perturbed state
    MultiFab trial(ba, dm, ncomp, nG);              // line-search trial
    MultiFab Ftrial(ba, dm, 1, 0);

    thd_diff::Workspace wsp;
    wsp.define(ba, dm);

    Precond pc;
    pc.define(ba, dm, geom);

    JacobianOp jop;
    jop.ba           = ba;
    jop.dm           = dm;
    jop.ncomp        = ncomp;
    jop.nghost_state = nG;
    jop.En     = &En;
    jop.Eh     = &Eh;
    jop.geom   = &geom;
    jop.pc     = &pc;
    jop.ws     = &wsp;
    jop.pert   = &pert;
    jop.base   = &base;
    jop.dt     = dt;
    jop.fd_eps = p.fd_eps;

    // Krylov sizing. The diagonal leaves cond(J) ~ 1 + dt*chi/dx^2
    // intact, so the basis is sized from the grid rather than left at
    // amrex's default restart length of 30.
    int restart = p.gmres_restart;
    if (restart <= 0) {
        const Box& dom = geom.Domain();
        int nmax = dom.length(0);
        for (int d = 1; d < AMREX_SPACEDIM; ++d) { nmax = amrex::max(nmax, dom.length(d)); }
        restart = amrex::max(amrex::min(2 * nmax, p.gmres_ceiling), 40);
    }
    const int maxits = (p.max_gmres > 0) ? p.max_gmres : restart;

    amrex::GMRES<MultiFab, JacobianOp> krylov;
    krylov.define(jop);
    krylov.setRestartLength(restart);
    krylov.setMaxIters(maxits);
    krylov.setVerbose(p.gmres_verbose);

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
        jop.prepare(state, F);

        // J dE = -F. amrex::GMRES zeroes the solution vector itself, so
        // there is no initial guess to supply.
        MultiFab::Copy(mrhs, F, 0, 0, 1, 0);
        mrhs.mult(-1.0_rt);
        krylov.solve(dE, mrhs, p.gmres_rtol, 0.0_rt);

        const int  nkr  = krylov.getNumIters();
        const Real kres = krylov.getResidualNorm() / amrex::max(Fnorm, 1.0e-300_rt);

        // A silent inner failure returns a partial vector and Newton
        // then follows a wrong direction with no indication. With a
        // diagonal preconditioner this is the expected failure mode as
        // dx shrinks: raise gmres_ceiling, or precondition properly.
        if (krylov.getStatus() != 0) {
            amrex::Print() << "[THD-JFNK] WARNING: GMRES stopped at " << nkr
                           << " iterations (status " << krylov.getStatus()
                           << ", basis " << restart << "), relative residual "
                           << kres << " > " << p.gmres_rtol
                           << ". Newton direction is inexact.\n";
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
            // converging solve turns into a diverging one. With an exact
            // Jacobian this should be rare; if it fires often, suspect an
            // under-converged Krylov solve first, then fd_eps (too small
            // and J*v is noise, too large and it is the wrong operator).
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
            amrex::Print() << "[THD-JFNK]   it=" << it
                           << "  krylov=" << nkr << "  kres=" << kres
                           << "  lambda=" << lambda << "  |F|=" << Fnorm
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
