// SimulationFields.cpp: sources of a slice, field solves, slice messages (part of the Simulation class)
#ifndef QUARZ_VERSION
#define QUARZ_VERSION "unknown"
#endif
#include "Simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "SimulationInternal.hpp"

namespace quarz {

using namespace detail;

// ---------------------------------------------------------------------------
void Simulation::zero_sources() {
    const int M = grid_->N + 1;
    SliceSources s = src_;
    Kokkos::parallel_for("zero_src", Range(0, M), KOKKOS_LAMBDA(int j) {
        const MScalar* sc[7] = {&s.rhot, &s.drho, &s.jz, &s.chi, &s.rho, &s.ne, &s.ni};
        for (int c = 0; c < 7; ++c) { sc[c]->a0(j) = 0; sc[c]->ar(j) = 0; sc[c]->ai(j) = 0; }
        const MVector* vc[2] = {&s.jp, &s.S};
        for (int c = 0; c < 2; ++c) {
            vc[c]->r0(j) = 0; vc[c]->i0(j) = 0; vc[c]->r1(j) = 0;
            vc[c]->i1(j) = 0; vc[c]->r2(j) = 0; vc[c]->i2(j) = 0;
        }
    });
}

void Simulation::combine_sources(int kl, Real ds) {
    const int M = grid_->N + 1;
    const int KL = box_.nloc;
    const int k = kl;
    SliceSources s = src_;
    MScalar bgt = bg_rhot_, bgr = bg_rho_;
    auto b = bsrc;
    auto gd = bguard;
    const Real idxi = Real(1) / box_.dxi;
    const bool m1 = m1_;
    const bool ngp = box_.ngp;
    const bool sub = ds != Real(0);
    Kokkos::parallel_for("combine", Range(0, M), KOKKOS_LAMBDA(int j) {
        // d(beam source)/dxi.  linear: central difference (serial only);
        // ngp: backward difference, the slice upstream of this rank comes from the guard row
        auto dxi_b = [&](int comp) {
            if (ngp) {
                const Real prev = (k > 0) ? b(k - 1, j, comp) : gd(j, comp);
                return (b(k, j, comp) - prev) * idxi;
            }
            if (KL == 1) return Real(0);
            if (k == 0) return (b(1, j, comp) - b(0, j, comp)) * idxi;
            if (k == KL - 1) return (b(KL - 1, j, comp) - b(KL - 2, j, comp)) * idxi;
            return Real(0.5) * (b(k + 1, j, comp) - b(k - 1, j, comp)) * idxi;
        };
        // beam source at xi_k (+ ds in a sub-slice: linear extrapolation with the same xi-derivative)
        auto bv = [&](int comp) { return sub ? b(k, j, comp) + ds * dxi_b(comp) : b(k, j, comp); };
        s.rhot.a0(j) += bgt.a0(j) + bv(B_RT0);
        s.drho.a0(j) += dxi_b(B_RT0);
        s.jz.a0(j) += bv(B_JZ0);
        s.jp.r1(j) += bv(B_JP1R);
        s.jp.i1(j) += bv(B_JP1I);
        s.rho.a0(j) += bgr.a0(j) + bv(B_RHO0);
        if (bgr.a0(j) > Real(0)) s.ni.a0(j) += bgr.a0(j);
        if (m1) {
            s.rhot.ar(j) += bgt.ar(j) + bv(B_RT1R);
            s.rhot.ai(j) += bgt.ai(j) + bv(B_RT1I);
            s.drho.ar(j) += dxi_b(B_RT1R);
            s.drho.ai(j) += dxi_b(B_RT1I);
            s.jz.ar(j) += bv(B_JZ1R);
            s.jz.ai(j) += bv(B_JZ1I);
            s.jp.r0(j) += bv(B_JP0R);
            s.jp.i0(j) += bv(B_JP0I);
            s.jp.r2(j) += bv(B_JP2R);
            s.jp.i2(j) += bv(B_JP2I);
            s.rho.ar(j) += bgr.ar(j) + bv(B_RHO1R);
            s.rho.ai(j) += bgr.ai(j) + bv(B_RHO1I);
        }
    });
}

void Simulation::set_guard(const double* p) {
    const int M = grid_->N + 1;
    auto h = Kokkos::create_mirror_view(bguard);
    for (int j = 0; j < M; ++j)
        for (int c = 0; c < B_NC1; ++c) h(j, c) = p[B_NC1 * j + c];
    Kokkos::deep_copy(bguard, h);
}

void Simulation::get_last_rhot(std::vector<double>& buf) const {
    const int M = grid_->N + 1;
    auto B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const int k = box_.nloc - 1;
    const int nc = static_cast<int>(B.extent(2));
    for (int j = 0; j < M; ++j)
        for (int c = 0; c < B_NC1; ++c) buf.push_back(c < nc ? B(k, j, c) : 0.0);
}

// W+ = -(d_x + i d_y) psi:  w1 = -psi0',  w2 = -(psi1' - psi1/r),  w0 = -(conj(psi1)' + conj(psi1)/r)
void Simulation::compute_wplus() {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    solver_->gradient(f.psi.a0, f.t1);
    const bool m1 = m1_;
    if (m1) {
        solver_->gradient_odd(f.psi.ar, f.t2);
        solver_->over_r(f.psi.ar, f.t3);
        solver_->gradient_odd(f.psi.ai, f.t4);
        solver_->over_r(f.psi.ai, f.rhs);
    }
    Kokkos::parallel_for("wplus", Range(0, M), KOKKOS_LAMBDA(int j) {
        f.wp.r1(j) = -f.t1(j);
        f.wp.i1(j) = Real(0);
        if (m1) {
            f.wp.r2(j) = -(f.t2(j) - f.t3(j));
            f.wp.i2(j) = -(f.t4(j) - f.rhs(j));
            f.wp.r0(j) = -(f.t2(j) + f.t3(j));
            f.wp.i0(j) = (f.t4(j) + f.rhs(j));
        }
    });
}

// L B_z = -curl_z J_perp;  curl_z = Im[(d_x - i d_y) J+],  (d_x - i d_y)(j_k e^{ik th}) = (j_k' + k j_k/r) e^{i(k-1) th}
void Simulation::compute_bz() {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    MVector g = gz_;   // scratch
    solver_->gradient_odd(s.jp.i1, f.t1);
    solver_->over_r(s.jp.i1, f.t2);
    const bool m1 = m1_;
    if (m1) {
        solver_->gradient_odd(s.jp.r2, g.r0);
        solver_->over_r(s.jp.r2, g.i0);
        solver_->gradient_odd(s.jp.i2, g.r1);
        solver_->over_r(s.jp.i2, g.i1);
        solver_->gradient(s.jp.r0, g.r2);
        solver_->gradient(s.jp.i0, g.i2);
    }
    Kokkos::parallel_for("bz_rhs", Range(0, M), KOKKOS_LAMBDA(int j) {
        f.t3(j) = -(f.t1(j) + f.t2(j));                         // -curl_z, mode 0
        if (m1) {
            const Real D1r = g.r0(j) + Real(2) * g.i0(j), D1i = g.r1(j) + Real(2) * g.i1(j);
            const Real Dmr = g.r2(j), Dmi = g.i2(j);
            // c1 = (D_1 - conj(D_-1)) / (2i)
            f.t4(j) = -Real(0.5) * (D1i + Dmi);                 // -Re c1
            f.rhs(j) = Real(0.5) * (D1r - Dmr);                 // -Im c1
        }
    });
    solver_->solve(OpKind::L0, f.t3, f.bz.a0);
    if (m1) {
        solver_->solve(OpKind::L1D, f.t4, f.bz.ar);
        solver_->solve(OpKind::L1D, f.rhs, f.bz.ai);
    }
}

// (L_k - chi0) b_k = i [ ((d_x + i d_y) J_z)_k + S_k ] + chi1 b_{k-1} + conj(chi1) b_{k+1}
void Simulation::solve_bplus(int /*k*/) {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    MVector g = gz_;
    const bool m1 = m1_;
    // ((d_x + i d_y) J_z):  k=1: J0' ;  k=2: J1' - J1/r ;  k=0: conj(J1)' + conj(J1)/r
    solver_->gradient(s.jz.a0, f.t1);
    if (m1) {
        solver_->gradient_odd(s.jz.ar, f.t2);
        solver_->over_r(s.jz.ar, f.t3);
        solver_->gradient_odd(s.jz.ai, f.t4);
        solver_->over_r(s.jz.ai, f.rhs);
    }
    Kokkos::parallel_for("gz", Range(0, M), KOKKOS_LAMBDA(int j) {
        g.r1(j) = f.t1(j) + s.S.r1(j);
        g.i1(j) = s.S.i1(j);
        if (m1) {
            g.r2(j) = (f.t2(j) - f.t3(j)) + s.S.r2(j);
            g.i2(j) = (f.t4(j) - f.rhs(j)) + s.S.i2(j);
            g.r0(j) = (f.t2(j) + f.t3(j)) + s.S.r0(j);
            g.i0(j) = -(f.t4(j) + f.rhs(j)) + s.S.i0(j);
        }
    });
    // rhs arrays reuse f.bprev (6 arrays)
    MVector rb = f.bprev;
    const int niter = m1 ? picard_ : 1;
    for (int it = 0; it < niter; ++it) {
        Kokkos::parallel_for("bplus_rhs", Range(0, M), KOKKOS_LAMBDA(int j) {
            // i G = (-G_i, G_r)
            Real r1 = -g.i1(j), i1 = g.r1(j);
            if (m1) {
                Real r0 = -g.i0(j), i0 = g.r0(j), r2 = -g.i2(j), i2 = g.r2(j);
                const Real cr = s.chi.ar(j), ci = s.chi.ai(j);   // chi1; conj(chi1) = (cr, -ci)
                const Real b0r = f.bp.r0(j), b0i = f.bp.i0(j), b1r = f.bp.r1(j), b1i = f.bp.i1(j);
                const Real b2r = f.bp.r2(j), b2i = f.bp.i2(j);
                // k=0: conj(chi1) b1
                r0 += cr * b1r + ci * b1i;  i0 += cr * b1i - ci * b1r;
                // k=1: chi1 b0 + conj(chi1) b2
                r1 += cr * b0r - ci * b0i + cr * b2r + ci * b2i;
                i1 += cr * b0i + ci * b0r + cr * b2i - ci * b2r;
                // k=2: chi1 b1
                r2 += cr * b1r - ci * b1i;  i2 += cr * b1i + ci * b1r;
                rb.r0(j) = r0; rb.i0(j) = i0; rb.r2(j) = r2; rb.i2(j) = i2;
            }
            rb.r1(j) = r1; rb.i1(j) = i1;
        });
        solver_->solve(OpKind::L1D, rb.r1, f.bp.r1, s.chi.a0);                 // B_r0
        solver_->solve(OpKind::L1F, rb.i1, f.bp.i1, s.chi.a0, s.jz.a0);       // B_theta0, flux wall
        if (m1) {
            solver_->solve(OpKind::L0, rb.r0, f.bp.r0, s.chi.a0);
            solver_->solve(OpKind::L0, rb.i0, f.bp.i0, s.chi.a0);
            solver_->solve(OpKind::L2D, rb.r2, f.bp.r2, s.chi.a0);
            solver_->solve(OpKind::L2D, rb.i2, f.bp.i2, s.chi.a0);
        }
    }
}

// E+ = W+ - i B+
void Simulation::store_slice(int kl) {
    const int k = kl;
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    auto F = fld;
    auto D = diag;
    const bool m1 = m1_;
    Kokkos::parallel_for("store", Range(0, M), KOKKOS_LAMBDA(int j) {
        F(k, j, F_EP1R) = f.wp.r1(j) + f.bp.i1(j);
        F(k, j, F_EP1I) = f.wp.i1(j) - f.bp.r1(j);
        F(k, j, F_BP1R) = f.bp.r1(j);
        F(k, j, F_BP1I) = f.bp.i1(j);
        F(k, j, F_EZ0) = f.ez.a0(j);
        F(k, j, F_BZ0) = f.bz.a0(j);
        D(k, j, D_PSI0) = f.psi.a0(j);
        D(k, j, D_NE0) = s.ne.a0(j);
        D(k, j, D_NI0) = s.ni.a0(j);
        if (m1) {
            F(k, j, F_EP0R) = f.wp.r0(j) + f.bp.i0(j);
            F(k, j, F_EP0I) = f.wp.i0(j) - f.bp.r0(j);
            F(k, j, F_EP2R) = f.wp.r2(j) + f.bp.i2(j);
            F(k, j, F_EP2I) = f.wp.i2(j) - f.bp.r2(j);
            F(k, j, F_BP0R) = f.bp.r0(j);
            F(k, j, F_BP0I) = f.bp.i0(j);
            F(k, j, F_BP2R) = f.bp.r2(j);
            F(k, j, F_BP2I) = f.bp.i2(j);
            F(k, j, F_EZ1R) = f.ez.ar(j);
            F(k, j, F_EZ1I) = f.ez.ai(j);
            F(k, j, F_BZ1R) = f.bz.ar(j);
            F(k, j, F_BZ1I) = f.bz.ai(j);
            D(k, j, D_PSI1R) = f.psi.ar(j);
            D(k, j, D_PSI1I) = f.psi.ai(j);
            D(k, j, D_NE1R) = s.ne.ar(j);
            D(k, j, D_NE1I) = s.ne.ai(j);
            D(k, j, D_NI1R) = s.ni.ar(j);
            D(k, j, D_NI1I) = s.ni.ai(j);
        }
    });
}

// msg: message from the upstream rank (empty on rank 0), already positioned after the beam part
void Simulation::step_fields(const std::vector<double>& msg, size_t& off) {
    const int K = box_.nxi, k0 = box_.k0, KL = box_.nloc;
    const int M = grid_->N + 1;

    if (laser_) laser_->begin_step();
    // ---- beams -> (local xi, r) arrays
    Kokkos::deep_copy(bsrc, 0.0);
    for (const auto& b : beams_) b->deposit(bsrc);
    if (imp_.extent(0) > 0) {
        Kokkos::deep_copy(imp_, 0.0);
        for (const auto& b : beams_) b->deposit_impact(imp_);
    }
    if (!ion_sp_.empty()) {
        Kokkos::deep_copy(diag, 0.0);
        for (auto& a : ion_acc_) a = {0, 0, 0, 0};
    }

    // ---- fresh plasma at the front of the box, background (every rank: the background is the same)
    const Real z_front = t_ - xi_head_;
    for (View1D v : {bg_rhot_.a0, bg_rhot_.ar, bg_rhot_.ai, bg_rho_.a0, bg_rho_.ar, bg_rho_.ai}) Kokkos::deep_copy(v, 0.0);
    unsigned long long sd = seed_;
    for (auto& s : species_) s->load(z_front, sd++);
    for (auto& s : species_)
        if (!s->mobile()) s->deposit_rho(bg_rhot_, bg_rho_);
    if (neutralize_) {
        MScalar a = src_.rhot, b = src_.rho;
        for (View1D v : {a.a0, a.ar, a.ai, b.a0, b.ar, b.ai}) Kokkos::deep_copy(v, 0.0);
        for (auto& s : species_)
            if (s->mobile()) s->deposit_rho(a, b);
        MScalar bgt = bg_rhot_, bgr = bg_rho_;
        Kokkos::parallel_for("neutralize", Range(0, grid_->N + 1), KOKKOS_LAMBDA(int j) {
            bgt.a0(j) -= a.a0(j); bgt.ar(j) -= a.ar(j); bgt.ai(j) -= a.ai(j);
            bgr.a0(j) -= b.a0(j); bgr.ar(j) -= b.ar(j); bgr.ai(j) -= b.ai(j);
        });
    }
    if (solver_->filter_on())   // the background is smoothed like the plasma (neutral plasma edge)
        for (int c = 0; c < 2; ++c) {
            MScalar& m = c == 0 ? bg_rhot_ : bg_rho_;
            solver_->filter(0, m.a0);
            if (m1_) { solver_->filter(1, m.ar); solver_->filter(1, m.ai); }
        }

    if (comm_.rank() == 0 && !bread_path_.empty()) {
        // stage 2: the plasma state leaving the stage-1 box, interpolated to this t; no beam upstream
        boundary_state(t_);
        size_t o = 0;
        for (auto& sp : species_)
            if (sp->mobile()) o += sp->unpack_state(bbuf_.data() + o);
        for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
            auto h = Kokkos::create_mirror_view(v);
            for (int j = 0; j < M; ++j) h(j) = bbuf_[o + j];
            Kokkos::deep_copy(v, h);
            o += M;
        }
        Kokkos::deep_copy(bguard, 0.0);
    } else if (comm_.rank() == 0) {
        // B+ initial guess for the Picard iteration: zero at the head; no beam upstream of the box
        for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) Kokkos::deep_copy(v, 0.0);
        Kokkos::deep_copy(bguard, 0.0);
    } else {
        // plasma state entering the first local slice, B+ warm start, upstream beam guard slice
        for (auto& sp : species_)
            if (sp->mobile()) off += sp->unpack_state(msg.data() + off);
        for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
            auto h = Kokkos::create_mirror_view(v);
            for (int j = 0; j < M; ++j) h(j) = msg[off + j];
            Kokkos::deep_copy(v, h);
            off += M;
        }
        set_guard(msg.data() + off);
        off += static_cast<size_t>(B_NC1) * M;
        if (laser_) {
            laser_->set_guard(msg.data() + off);
            off += Laser::guard_size(M);
        }
    }

    const Real dxi = box_.dxi;
    for (int kl = 0; kl < KL; ++kl) {
        const int k = k0 + kl;   // global slice
        zero_sources();
        if (laser_) laser_->prepare_slice(kl, fld_);   // <a^2> of this slice (time n)
        for (auto& sp : species_)
            if (sp->mobile()) sp->deposit(src_, fld_);
        filter_plasma_sources();
        if (laser_) laser_->advance_slice(kl, src_.chi.a0);   // envelope of this slice -> time n+1
        combine_sources(kl);
        solve_slice_fields();
        store_slice(kl);
        for (size_t s = 0; s < ion_sp_.size(); ++s) {   // ionization with the fields of this slice
            PlasmaSpecies& ion = *species_[ion_sp_[s]];
            const IonizeResult r = ion.ionize(fld_, imp_, kl, dxi, k, step_, *species_[ion_prod_[s]]);
            ion_acc_[s][0] += r.born_w;
            ion_acc_[s][1] += r.born_wp2;
            ion.deposit_charge_state(diag, kl, diag_base_ + static_cast<int>(s));
        }
        if (k < K - 1 || !bwrite_path_.empty()) {   // the last local push produces the state entering the next rank
            // (and, with boundary.write, the state leaving the box)
            // adaptive sub-slicing: if plasma particles would cross more than max_cells_ radial cells
            // in this step, it is split into nsub sub-slices, each with its own deposit and field
            // solve (beam rho - J_z extrapolated from this slice, laser <a^2> and ionization of this
            // slice); only the regular slices are stored
            const int nsub = subslices(dxi);
            const Real h = dxi / nsub;
            for (int ss = 0; ss < nsub; ++ss) {
                if (ss > 0) {
                    zero_sources();
                    for (auto& sp : species_)
                        if (sp->mobile()) sp->deposit(src_, fld_);
                    filter_plasma_sources();
                    combine_sources(kl, ss * h);
                    solve_slice_fields();
                }
                for (auto& sp : species_)
                    if (sp->mobile()) sp->push(fld_, h);
            }
            nsub_step_ += nsub - 1;
        }
    }
    if (laser_) laser_->end_sweep();
    if (k0 + KL == K)   // the last rank: plasma state at the end of the box
        for (size_t s = 0; s < ion_sp_.size(); ++s) species_[ion_sp_[s]]->charge_sums(ion_acc_[s][2], ion_acc_[s][3]);
    Kokkos::fence();
}

void Simulation::solve_slice_fields() {
    solver_->solve(OpKind::L0, src_.rhot.a0, fld_.psi.a0, View1D(), View1D(), -1.0);
    solver_->solve(OpKind::L0, src_.drho.a0, fld_.ez.a0, View1D(), View1D(), -1.0);
    if (m1_) {
        solver_->solve(OpKind::L1D, src_.rhot.ar, fld_.psi.ar, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.rhot.ai, fld_.psi.ai, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.drho.ar, fld_.ez.ar, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.drho.ai, fld_.ez.ai, View1D(), View1D(), -1.0);
    }
    compute_wplus();
    compute_bz();
    for (auto& sp : species_)
        if (sp->mobile()) sp->deposit_S(fld_, src_.S);
    if (solver_->filter_on()) {
        const MVector& S = src_.S;
        solver_->filter(1, S.r1); solver_->filter(1, S.i1);
        if (m1_) {
            solver_->filter(0, S.r0); solver_->filter(0, S.i0);
            solver_->filter(2, S.r2); solver_->filter(2, S.i2);
        }
    }
    solve_bplus(0);
}

// plasma deposit only (beams and the background are added later in combine_sources; the background
// is filtered once per step). Linear and xi-independent, so E_z = d psi / d xi stays exact.
void Simulation::filter_plasma_sources() {
    if (!solver_->filter_on()) return;
    const SliceSources& s = src_;
    for (const MScalar* m : {&s.rhot, &s.drho, &s.jz, &s.chi, &s.rho, &s.ne, &s.ni}) {
        solver_->filter(0, m->a0);
        if (m1_) { solver_->filter(1, m->ar); solver_->filter(1, m->ai); }
    }
    solver_->filter(1, s.jp.r1); solver_->filter(1, s.jp.i1);
    if (m1_) {
        solver_->filter(0, s.jp.r0); solver_->filter(0, s.jp.i0);
        solver_->filter(2, s.jp.r2); solver_->filter(2, s.jp.i2);
    }
}

int Simulation::subslices(Real dxi) const {
    if (!(max_cells_ > 0)) return 1;
    Real c = 0;
    double hlast = 0;
    for (const auto& sp : species_)
        if (sp->mobile()) { c = std::max(c, sp->max_cells(dxi)); hlast = std::max(hlast, sp->last_step()); }
    int n = c <= max_cells_ ? 1 : static_cast<int>(std::ceil(c / max_cells_));
    // the step may at most double: at least ceil(dxi / (2 h_last)) sub-slices after finer steps
    if (hlast > 0) n = std::max(n, static_cast<int>(std::ceil(dxi / (2 * hlast) - 1e-9)));
    return std::min(substep_max_, n);
}

// message to the downstream rank for step n:
//   [mobile species states (incl. AB history positions)][B+ (6 M)][beam sources of the last local slice (15 M)][beam particles]
std::vector<double> Simulation::make_message(int n) {
    std::vector<double> m;
    m.push_back(static_cast<double>(n));
    m.push_back(cur_dt_);
    m.push_back(cur_last_ ? 1.0 : 0.0);
    for (size_t b = 0; b < beams_.size(); ++b) {
        m.push_back(static_cast<double>(outbox_[b].size() / beams_[b]->ncomp()));
        m.insert(m.end(), outbox_[b].begin(), outbox_[b].end());
        outbox_[b].clear();
    }
    for (auto& sp : species_)
        if (sp->mobile()) sp->pack_state(m);
    for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), v);
        m.insert(m.end(), h.data(), h.data() + h.extent(0));
    }
    get_last_rhot(m);
    if (laser_) laser_->get_guard(m);
    return m;
}

} // namespace quarz
