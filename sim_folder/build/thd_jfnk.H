#ifndef THD_JFNK_H
#define THD_JFNK_H

#include <thd_diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_GMRES.H>
#include <vector>
#include <string>
#include <memory>

using namespace amrex;

namespace thd_jfnk {

enum DiffMode { MODE_OFF = 0, MODE_IMPLICIT = 1, MODE_MG = 2 };
enum PrecondType { PC_JACOBI = 0, PC_MG = 1 };

struct Params {
    DiffMode    mode    = MODE_MG;
    PrecondType precond = PC_MG;

    bool use_cache = true;
    Real source_scale = 1.0_rt;

    int  max_newton     = 20;
    Real newton_rtol    = 1.0e-6_rt;
    int  max_linesearch = 8;
    Real fd_eps         = 1.0e-7_rt;

    int  max_gmres      = 0;
    int  gmres_ceiling  = 120;
    Real gmres_rtol     = 1.0e-4_rt;
    Real reorth_tol     = 0.7_rt;

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

    bool verbose = false;
};

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

struct Precond {

    PrecondType type = PC_MG;
    bool        defined = false;

    MultiFab diag;
    MultiFab alpha;
    MultiFab wsol;
    MultiFab rhs;
    std::unique_ptr<MLABecLaplacian> op;
    std::unique_ptr<MLMG>            mlmg;
    Geometry geom;
    Real     dt_cur = 0.0_rt;
    Params   par;

    bool need_mg () const { return type == PC_MG; }

    void define (const BoxArray& ba, const DistributionMapping& dm,
                 const Geometry& gm, const Params& p, bool mg_needed)
    {
        par     = p;
        geom    = gm;
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

        Array<LinOpBCType,AMREX_SPACEDIM> lobc, hibc;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (gm.isPeriodic(idim)) {
                lobc[idim] = hibc[idim] = LinOpBCType::Periodic;
            } else {
                lobc[idim] = hibc[idim] = LinOpBCType::Neumann;
            }
        }
        op->setDomainBC(lobc, hibc);
        op->setMaxOrder(2);
        op->setLevelBC(0, nullptr);
    }

    void update (const MultiFab& state, const MultiFab& Eh,
                 Real dt, thd_diff::Workspace* ws)
    {
        dt_cur = dt;

        thd_diff::compute_coeffs(state, ws->coef);
        thd_diff::compute_face_coeffs(ws->coef, ws->fk, geom);

        if (!need_mg()) { build_diag(state, Eh, dt, ws); return; }

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

        op->setScalars(1.0_rt, dt);
        op->setACoeffs(0, alpha);
        op->setBCoeffs(0, Array<MultiFab const*,AMREX_SPACEDIM>{
                              AMREX_D_DECL(&ws->fk[0], &ws->fk[1], &ws->fk[2])});

        mlmg = std::make_unique<MLMG>(*op);
        mlmg->setVerbose(par.mg_verbose);
        mlmg->setBottomVerbose(par.mg_bottom_verbose);
        mlmg->setPreSmooth(par.mg_pre_smooth);
        mlmg->setPostSmooth(par.mg_post_smooth);
        if (par.mg_bottom_smoother) {
            mlmg->setBottomSolver(MLMG::BottomSolver::smoother);
        }
    }

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
        MultiFab::Multiply(x, alpha, 0, 0, 1, 0);
    }

    Real newton_direction (const MultiFab& F, MultiFab& dE)
    {
        AMREX_ALWAYS_ASSERT(need_mg());

        MultiFab::Copy(rhs, F, 0, 0, 1, 0);
        rhs.mult(-1.0_rt);
        wsol.setVal(0.0_rt);

        mlmg->setFixedIter(0);
        mlmg->setMaxIter(par.mg_max_iter);
        const Real res = mlmg->solve(Vector<MultiFab*>{&wsol},
                                     Vector<MultiFab const*>{&rhs},
                                     par.mg_rtol, par.mg_atol);

        MultiFab::Copy(dE, wsol, 0, 0, 1, 0);
        MultiFab::Multiply(dE, alpha, 0, 0, 1, 0);
        return res;
    }

    void build_diag (const MultiFab& state, const MultiFab& Eh,
                     Real dt, thd_diff::Workspace* ws)
    {
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

// ====================================================================
// Matrix-Free Linear Operator Adaptor for amrex::GMRES
// ====================================================================
struct JFNKLinOp
{
    const MultiFab* state_ptr = nullptr;
    const MultiFab* F0_ptr    = nullptr;
    const MultiFab* En_ptr    = nullptr;
    const MultiFab* Eh_ptr    = nullptr;
    Precond*        pc_ptr    = nullptr;
    const Geometry* geom_ptr  = nullptr;
    Real            dt        = 0.0_rt;
    const Params*   p_ptr     = nullptr;
    thd_diff::Workspace* ws   = nullptr;
    MultiFab*       pert_ptr  = nullptr;
    MultiFab*       base_ptr  = nullptr;
    MultiFab*       w_ptr     = nullptr;

    // AMReX GMRES Matrix-Vector Product Interface: y = J * x
    void apply (MultiFab& y, const MultiFab& x) const
    {
        const Real norm_E = state_ptr->norm2(UEINT);
        const Real norm_v = x.norm2(0);
        const Real eps    = p_ptr->fd_eps * (norm_E + 1.0_rt) / (norm_v + 1.0e-300_rt);

        apply_update(*pert_ptr, *base_ptr, x, eps, *Eh_ptr, *geom_ptr);
        residual(*pert_ptr, *En_ptr, *w_ptr, *geom_ptr, dt, ws);
        MultiFab::Saxpy(*w_ptr, -1.0_rt, *F0_ptr, 0, 0, 1, 0);
        w_ptr->mult(1.0_rt / eps);

        MultiFab::Copy(y, *w_ptr, 0, 0, 1, 0);
    }

    // AMReX GMRES Preconditioner Interface: y = M^-1 * x
    void applyPrecond (MultiFab& y, const MultiFab& x) const
    {
        MultiFab::Copy(y, x, 0, 0, 1, 0);
        pc_ptr->apply_inplace(y);
    }
};

// Left-preconditioned GMRES wrapper using amrex::GMRES.
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

    MultiFab::Copy(pert, state, 0, 0, state.nComp(), state.nGrow());
    MultiFab::Copy(base, state, UEINT, 0, 1, 0);

    // Prepare RHS: b = -F0
    MultiFab b(ba, dm, 1, 0);
    MultiFab::Copy(b, F0, 0, 0, 1, 0);
    b.mult(-1.0_rt);

    int m = p.max_gmres;
    if (m <= 0) {
        if (pc.need_mg()) {
            m = amrex::min(40, p.gmres_ceiling);
        } else {
            const Box& dom = geom.Domain();
            int nmax = dom.length(0);
            for (int d = 1; d < AMREX_SPACEDIM; ++d) { nmax = amrex::max(nmax, dom.length(d)); }
            m = amrex::max(amrex::min(2 * nmax, p.gmres_ceiling), 40);
        }
    }

    JFNKLinOp linop;
    linop.state_ptr = &state;
    linop.F0_ptr    = &F0;
    linop.En_ptr    = &En;
    linop.Eh_ptr    = &Eh;
    linop.pc_ptr    = &pc;
    linop.geom_ptr  = &geom;
    linop.dt        = dt;
    linop.p_ptr     = &p;
    linop.ws        = ws;
    linop.pert_ptr  = &pert;
    linop.base_ptr  = &base;
    linop.w_ptr     = &w;

    // Construct AMReX GMRES solver
    amrex::GMRES<MultiFab, JFNKLinOp> gmres_solver(linop);
    gmres_solver.setMaxIters(m);
    gmres_solver.setVerbose(p.verbose ? 1 : 0);

    // Execute solve: J * dE = b
    int ret = gmres_solver.solve(dE, b, p.gmres_rtol, 0.0_rt);

    if (final_resid != nullptr) {
        *final_resid = gmres_solver.getResidualNorm();
    }

    return ret;
}

} // namespace thd_jfnk

#endif // THD_JFNK_H