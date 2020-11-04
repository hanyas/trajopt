#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace py = pybind11;

using namespace arma;


typedef py::array_t<double, py::array::f_style | py::array::forcecast> array_tf;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> array_tc;


cube array_to_cube(array_tf m) {

    py::buffer_info _m_buff = m.request();
    int n_rows = _m_buff.shape[0];
    int n_cols = _m_buff.shape[1];
    int n_slices = _m_buff.shape[2];

    cube _m_arma((double *)_m_buff.ptr, n_rows, n_cols, n_slices);

    return _m_arma;
}


mat array_to_mat(array_tf m) {

    py::buffer_info _m_buff = m.request();
    int n_rows = _m_buff.shape[0];
    int n_cols = _m_buff.shape[1];

    mat _m_arma((double *)_m_buff.ptr, n_rows, n_cols);

    return _m_arma;
}


vec array_to_vec(array_tf m) {

    py::buffer_info _m_buff = m.request();
    int n_rows = _m_buff.shape[0];

    vec _m_vec((double *)_m_buff.ptr, n_rows);

    return _m_vec;
}


array_tf cube_to_array(cube m) {

    auto _m_array = array_tf({m.n_rows, m.n_cols, m.n_slices});

    py::buffer_info _m_buff = _m_array.request();
    std::memcpy(_m_buff.ptr, m.memptr(), sizeof(double) * m.n_rows * m.n_cols * m.n_slices);

    return _m_array;
}


array_tf mat_to_array(mat m) {

    auto _m_array = array_tf({m.n_rows, m.n_cols});

    py::buffer_info _m_buff = _m_array.request();
    std::memcpy(_m_buff.ptr, m.memptr(), sizeof(double) * m.n_rows * m.n_cols);

    return _m_array;
}


array_tf vec_to_array(vec m) {

    auto _m_array = array_tf({m.n_rows});

    py::buffer_info _m_buff = _m_array.request();
    std::memcpy(_m_buff.ptr, m.memptr(), sizeof(double) * m.n_rows);

    return _m_array;
}


double kl_divergence(array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                       array_tf _lK, array_tf _lkff, array_tf _lsigma_ctl,
                       array_tf _mu_x, array_tf _sigma_x,
                       int dm_state, int dm_act, int nb_steps) {

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    cube lK = array_to_cube(_lK);
    mat lkff = array_to_mat(_lkff);
    cube lsigma_ctl = array_to_cube(_lsigma_ctl);

    mat mu_x  = array_to_mat(_mu_x);
    cube sigma_x = array_to_cube(_sigma_x);

    double kl = 0.0;

    for(int i = 0; i < nb_steps; i++) {
        mat lprec_ctl = inv_sympd(lsigma_ctl.slice(i));

        mat diff_K = (lK.slice(i) - K.slice(i)).t() * lprec_ctl * (lK.slice(i) - K.slice(i));
        mat diff_crs = (lK.slice(i) - K.slice(i)).t() * lprec_ctl * (- lkff.col(i) + kff.col(i));
        mat diff_kff = (- lkff.col(i) + kff.col(i)).t() * lprec_ctl * (- lkff.col(i) + kff.col(i));

        kl += as_scalar(0.5 * log( det(lsigma_ctl.slice(i)) / det(sigma_ctl.slice(i)) )
		                + 0.5 * trace(lprec_ctl * sigma_ctl.slice(i))
		                - 0.5 * dm_act
		                + 0.5 * trace(diff_K * sigma_x.slice(i))
		                + 0.5 * mu_x.col(i).t() * diff_K * mu_x.col(i)
		                - mu_x.col(i).t() * diff_crs
		                + 0.5 * diff_kff);
    }

    return kl;
}

double quad_expectation(array_tf _mu, array_tf _sigma_s,
                        array_tf _Q, array_tf _q, double _q0) {

    vec mu  = array_to_vec(_mu);
    mat sigma_s = array_to_mat(_sigma_s);

    mat Q = array_to_mat(_Q);
    vec q = array_to_vec(_q);

	double result = as_scalar(mu.t() * Q * mu) + as_scalar(mu.t() * q) + _q0 + trace(Q * sigma_s);
	return result;
}

py::tuple augment_cost(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
                       array_tf _cu, array_tf _Cxu, array_tf _c0,
                       array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                       double alpha, int dm_state, int dm_act, int nb_steps) {

    // inputs
    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    // outputs
    cube agCxx(dm_state, dm_state, nb_steps + 1);
    mat agcx(dm_state, nb_steps + 1);
    cube agCuu(dm_act, dm_act, nb_steps + 1);
    mat agcu(dm_act, nb_steps + 1);
    cube agCxu(dm_state, dm_act, nb_steps + 1);
    vec agc0(nb_steps + 1);

    for (int i = 0; i < nb_steps; i++) {
        mat prec_ctl = inv_sympd(sigma_ctl.slice(i));

        agCxx.slice(i) = Cxx.slice(i) - 0.5 * alpha * K.slice(i).t() * prec_ctl * K.slice(i);
        agCuu.slice(i) = Cuu.slice(i) - 0.5 * alpha * prec_ctl;
        agCxu.slice(i) = Cxu.slice(i) + 0.5 * alpha * K.slice(i).t() * prec_ctl;
        agcx.col(i) = cx.col(i) - alpha * K.slice(i).t() * prec_ctl * kff.col(i);
        agcu.col(i) = cu.col(i) + alpha * prec_ctl * kff.col(i);
        agc0(i) = as_scalar(c0(i) - 0.5 * alpha * log( det(2. * datum::pi * sigma_ctl.slice(i)) )
                   - 0.5 * alpha * kff.col(i).t() * prec_ctl * kff.col(i));
    }

    // last time step
    agCxx.slice(nb_steps) = Cxx.slice(nb_steps);
    agcx.col(nb_steps) = cx.col(nb_steps);
    agCuu.slice(nb_steps) = Cuu.slice(nb_steps);
    agcu.col(nb_steps) = cu.col(nb_steps);
    agCxu.slice(nb_steps) = Cxu.slice(nb_steps);
    agc0(nb_steps) = c0(nb_steps);

    // transform outputs to numpy
    array_tf _agCxx = cube_to_array(agCxx);
    array_tf _agcx = mat_to_array(agcx);
    array_tf _agCuu =  cube_to_array(agCuu);
    array_tf _agcu = mat_to_array(agcu);
    array_tf _agCxu =  cube_to_array(agCxu);
    array_tf _agc0 = vec_to_array(agc0);

    py::tuple output =  py::make_tuple(_agCxx, _agcx, _agCuu, _agcu, _agCxu, _agc0);
    return output;
}

py::tuple forward_pass(array_tf _mu_x0, array_tf _sigma_x0,
                       array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_dyn,
                       array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                       int dm_state, int dm_act, int nb_steps) {

    // inputs
    vec mu_x0 = array_to_vec(_mu_x0);
    mat sigma_x0 = array_to_mat(_sigma_x0);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);
    mat c = array_to_mat(_c);
    cube sigma_dyn = array_to_cube(_sigma_dyn);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    // outputs
    mat mu_x(dm_state, nb_steps + 1);
    cube sigma_x(dm_state, dm_state, nb_steps + 1);

    mat mu_u(dm_act, nb_steps);
    cube sigma_u(dm_act, dm_act, nb_steps);

    mat mu_xu(dm_state + dm_act, nb_steps + 1);
    cube sigma_xu(dm_state + dm_act, dm_state + dm_act, nb_steps + 1);

    mu_x.col(0) = mu_x0;
    sigma_x.slice(0) = sigma_x0;

    for (int i = 0; i < nb_steps; i++) {

        // mu_u = K * mu_x + k
        mu_u.col(i) = K.slice(i) * mu_x.col(i) + kff.col(i);

        // sigma_u = sigma_ctl + K * sigma_x * K_T
        sigma_u.slice(i) = sigma_ctl.slice(i) + K.slice(i) * sigma_x.slice(i) * K.slice(i).t();
        sigma_u.slice(i) = 0.5 * (sigma_u.slice(i) + sigma_u.slice(i).t());

        // sigma_xu =   [[sigma_x,      sigma_x * K_T],
        //               [K*sigma_x,    sigma_u    ]]
        sigma_xu.slice(i) = join_vert(join_horiz(sigma_x.slice(i), sigma_x.slice(i) * K.slice(i).t()),
                                        join_horiz(K.slice(i) * sigma_x.slice(i), sigma_u.slice(i)));
        sigma_xu.slice(i) = 0.5 * (sigma_xu.slice(i) + sigma_xu.slice(i).t());

        // mu_xu =  [[mu_x],
        //           [mu_u]],
        mu_xu.col(i) = join_vert(mu_x.col(i), mu_u.col(i));

        // sigma_x_next = sigma_dyn + [A B] * sigma_xu * [A B]^T
        sigma_x.slice(i+1) = sigma_dyn.slice(i) + join_horiz(A.slice(i), B.slice(i)) * sigma_xu.slice(i) *
                                                              join_vert(A.slice(i).t(), B.slice(i).t());
        sigma_x.slice(i+1) = 0.5 * (sigma_x.slice(i+1) + sigma_x.slice(i+1).t());

        // mu_x_next = [A B] * [s a]^T + c
        mu_x.col(i+1) = join_horiz(A.slice(i), B.slice(i)) * mu_xu.col(i) + c.col(i);

        if(i == nb_steps - 1) {
            mu_xu.col(i+1) = join_vert(mu_x.col(i+1), zeros<vec>(dm_act));
            sigma_xu.slice(i+1).submat(0, 0, dm_state - 1, dm_state - 1) = sigma_x.slice(i+1);
        }
    }

    // transform outputs to numpy
    array_tf _mu_x = mat_to_array(mu_x);
    array_tf _sigma_x = cube_to_array(sigma_x);
    array_tf _mu_u =  mat_to_array(mu_u);
    array_tf _sigma_u = cube_to_array(sigma_u);
    array_tf _mu_xu =  mat_to_array(mu_xu);
    array_tf _sigma_xu = cube_to_array(sigma_xu);

    py::tuple output =  py::make_tuple(_mu_x, _sigma_x, _mu_u, _sigma_u, _mu_xu, _sigma_xu);
    return output;
}


py::tuple cubature_forward_pass(array_tf _mu_x0, array_tf _sigma_x0,
                       array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_param,
                       array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                       int dm_state, int dm_act, int nb_steps) {

    // inputs
    vec mu_x0 = array_to_vec(_mu_x0);
    mat sigma_x0 = array_to_mat(_sigma_x0);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);
    mat c = array_to_mat(_c);

    cube sigma_param = array_to_cube(_sigma_param);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    // augmented state dimension
    int dm_state_aug = 2*dm_state + dm_act + 1;
    mat chol_sigma_aug(dm_state_aug, dm_state_aug);
    vec mu_aug(dm_state_aug);

    // cubature
    mat cubature_points(dm_state_aug, 2*dm_state_aug);
    mat prop_cubature_points(dm_state, 2*dm_state_aug);
    mat sigma_all(dm_state, dm_state);

    // outputs
    mat mu_x(dm_state, nb_steps + 1);
    cube sigma_x(dm_state, dm_state, nb_steps + 1);

    mat mu_u(dm_act, nb_steps);
    cube sigma_u(dm_act, dm_act, nb_steps);

    mat mu_xu(dm_state + dm_act, nb_steps + 1);
    cube sigma_xu(dm_state + dm_act, dm_state + dm_act, nb_steps + 1);

    mu_x.col(0) = mu_x0;
    sigma_x.slice(0) = sigma_x0;

    for (int i = 0; i < nb_steps; i++) {

        // mu_u = K * mu_x + k
        mu_u.col(i) = K.slice(i) * mu_x.col(i) + kff.col(i);

        // sigma_u = sigma_ctl + K * sigma_x * K_T
        sigma_u.slice(i) = sigma_ctl.slice(i) + K.slice(i) * sigma_x.slice(i) * K.slice(i).t();
        sigma_u.slice(i) = 0.5 * (sigma_u.slice(i) + sigma_u.slice(i).t());

        // sigma_xu =   [[sigma_x,      sigma_x * K_T],
        //               [K*sigma_x,    sigma_u    ]]
        sigma_xu.slice(i) = join_vert(join_horiz(sigma_x.slice(i), sigma_x.slice(i) * K.slice(i).t()),
                                        join_horiz(K.slice(i) * sigma_x.slice(i), sigma_u.slice(i)));
        sigma_xu.slice(i) = 0.5 * (sigma_xu.slice(i) + sigma_xu.slice(i).t());

        // mu_xu =  [[mu_x],
        //           [mu_u]],
        mu_xu.col(i) = join_vert(mu_x.col(i), mu_u.col(i));

        // form augmented state mean and covariance
        mu_aug = join_vert(mu_xu.col(i), ones(1), zeros<vec>(dm_state));

        chol_sigma_aug.fill(0.0);
        chol_sigma_aug.submat(0, 0, dm_state + dm_act - 1, dm_state + dm_act - 1) = chol(sigma_xu.slice(i), "lower");
        chol_sigma_aug.submat(dm_state + dm_act + 1, dm_state + dm_act + 1, dm_state_aug - 1, dm_state_aug - 1) = eye(dm_state, dm_state);
        
        // calculate cubature points
        cubature_points = join_horiz(chol_sigma_aug, -chol_sigma_aug);
        cubature_points *= sqrt(dm_state_aug);
        cubature_points.each_col() += mu_aug;

        // propagate cubature points
         for (int j = 0; j < 2*dm_state_aug; j++){
            vec y = cubature_points.col(j);
            sigma_all = kron(y(span(0, dm_state + dm_act)).t(), eye(dm_state,dm_state))*sigma_param.slice(i)*kron(y(span(0, dm_state + dm_act)), eye(dm_state,dm_state));
            prop_cubature_points.col(j) = join_horiz(A.slice(i), B.slice(i), c.col(i), chol(sigma_all, "lower"))*y;
        } 
        
        // estimate new mean and covariance
        mu_x.col(i+1) = mean(prop_cubature_points, 1);
        
        prop_cubature_points.each_col() -= mu_x.col(i+1);

        sigma_x.slice(i+1).fill(0.0);
        prop_cubature_points.each_col([&sigma_x, i] (vec& col){
            sigma_x.slice(i+1) += col*col.t();
        });
        sigma_x.slice(i+1) /= 2*dm_state_aug;
        sigma_x.slice(i+1) = 0.5 * (sigma_x.slice(i+1) + sigma_x.slice(i+1).t());


        if(i == nb_steps - 1) {
            mu_xu.col(i+1) = join_vert(mu_x.col(i+1), zeros<vec>(dm_act));
            sigma_xu.slice(i+1).submat(0, 0, dm_state - 1, dm_state - 1) = sigma_x.slice(i+1);
        }
    }

    // transform outputs to numpy
    array_tf _mu_x = mat_to_array(mu_x);
    array_tf _sigma_x = cube_to_array(sigma_x);
    array_tf _mu_u =  mat_to_array(mu_u);
    array_tf _sigma_u = cube_to_array(sigma_u);
    array_tf _mu_xu =  mat_to_array(mu_xu);
    array_tf _sigma_xu = cube_to_array(sigma_xu);

    py::tuple output =  py::make_tuple(_mu_x, _sigma_x, _mu_u, _sigma_u, _mu_xu, _sigma_xu);
    return output;
}


py::tuple backward_pass(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
                        array_tf _cu, array_tf _Cxu, array_tf _c0,
                        array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_dyn,
                        double alpha, int dm_state, int dm_act, int nb_steps) {

    // inputs
    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);
    mat c = array_to_mat(_c);
    cube sigma_dyn = array_to_cube(_sigma_dyn);

    // outputs
    cube Q(dm_state + dm_act, dm_state + dm_act, nb_steps);
    cube Qxx(dm_state, dm_state, nb_steps);
    cube Qux(dm_act, dm_state, nb_steps);
    cube Quu(dm_act, dm_act, nb_steps);
    cube Quu_inv(dm_act, dm_act, nb_steps);
    mat qx(dm_state, nb_steps);
    mat qu(dm_act, nb_steps);
    vec q0(nb_steps);
    vec q0_common(nb_steps);
    vec q0_softmax(nb_steps);

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);
    vec v0_softmax(nb_steps + 1);

    cube K(dm_act, dm_state, nb_steps);
    mat kff(dm_act, nb_steps);
    cube sigma_ctl(dm_act, dm_act, nb_steps);
    cube prec_ctl(dm_act, dm_act, nb_steps);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = Cxx.slice(nb_steps);
    v.col(nb_steps) = cx.col(nb_steps);
    v0(nb_steps) = c0(nb_steps);
    v0_softmax(nb_steps) = c0(nb_steps);

	for(int i = nb_steps - 1; i>= 0; --i)
	{
        Qxx.slice(i) = (Cxx.slice(i) + A.slice(i).t() * V.slice(i+1) * A.slice(i)) / alpha;
        Quu.slice(i) = (Cuu.slice(i) + B.slice(i).t() * V.slice(i+1) * B.slice(i)) / alpha;
        Qux.slice(i) = (Cxu.slice(i) + A.slice(i).t() * V.slice(i+1) * B.slice(i)).t() / alpha;

        qu.col(i) = (cu.col(i) + 2.0 * B.slice(i).t() * V.slice(i+1) * c.col(i) + B.slice(i).t() * v.col(i+1)) / alpha;
        qx.col(i) = (cx.col(i) + 2.0 * A.slice(i).t() * V.slice(i+1) * c.col(i) + A.slice(i).t() * v.col(i+1)) / alpha;
        q0_common(i) = as_scalar(c0(i) +  c.col(i).t() * V.slice(i+1) * c.col(i)
                        + trace(V.slice(i+1) * sigma_dyn.slice(i)) + v.col(i+1).t() * c.col(i));

        q0(i) = (q0_common(i) + v0(i+1)) / alpha;
        q0_softmax(i) = (q0_common(i) + v0_softmax(i+1)) / alpha;

        if ((Quu.slice(i)).is_sympd()) {
            _diverge = i;
            break;
        }

        Quu_inv.slice(i) = inv(Quu.slice(i));
        K.slice(i) = - Quu_inv.slice(i) * Qux.slice(i);
        kff.col(i) = - 0.5 * Quu_inv.slice(i) * qu.col(i);

        sigma_ctl.slice(i) = - 0.5 * Quu_inv.slice(i);
        sigma_ctl.slice(i) = 0.5 * (sigma_ctl.slice(i).t() + sigma_ctl.slice(i));

        prec_ctl.slice(i) = - (Quu.slice(i).t() + Quu.slice(i));
        prec_ctl.slice(i) = 0.5 * (prec_ctl.slice(i).t() + prec_ctl.slice(i));

        V.slice(i) = (Qxx.slice(i) + Qux.slice(i).t() * K.slice(i)) * alpha;
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = (qx.col(i) + 2. * Qux.slice(i).t() * kff.col(i)) * alpha;
        v0(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i) - (0.5 * dm_act));
        v0_softmax(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0_softmax(i)
                         + 0.5 * (dm_act * log (2. * datum::pi) - log(det(- 2. * Quu.slice(i)))));
	}

    // transform outputs to numpy
    array_tf _Qxx = cube_to_array(Qxx);
    array_tf _Qux = cube_to_array(Qux);
    array_tf _Quu = cube_to_array(Quu);

    array_tf _qx = mat_to_array(qx);
    array_tf _qu = mat_to_array(qu);
    array_tf _q0 = mat_to_array(q0);
    array_tf _q0_softmax = mat_to_array(q0_softmax);

    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);
    array_tf _v0_softmax = vec_to_array(v0_softmax);

    array_tf _K = cube_to_array(K);
    array_tf _kff = mat_to_array(kff);
    array_tf _sigma_ctl = cube_to_array(sigma_ctl);

    py::tuple output =  py::make_tuple(_Qxx, _Qux, _Quu, _qx, _qu, _q0, _q0_softmax,
                                        _V, _v, _v0, _v0_softmax,
                                        _K, _kff, _sigma_ctl, _diverge);

    return output;
}

py::tuple robust_backward_pass(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
                        array_tf _cu, array_tf _Cxu, array_tf _c0,
                        array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_param,
                        double alpha, int dm_state, int dm_act, int nb_steps) {

    // inputs
    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);
    mat c = array_to_mat(_c);
    cube sigma_param = array_to_cube(_sigma_param);   

    //
    mat P(dm_state + dm_act + 1, dm_state + dm_act + 1);
    mat Pxx(dm_state, dm_state);
    mat Pxu(dm_act, dm_state);
    mat Puu(dm_act, dm_act);
    vec px(dm_state);
    vec pu(dm_act);
    double p0;
    
    // outputs
    cube Q(dm_state + dm_act, dm_state + dm_act, nb_steps);
    cube Qxx(dm_state, dm_state, nb_steps);
    cube Qux(dm_act, dm_state, nb_steps);
    cube Quu(dm_act, dm_act, nb_steps);
    cube Quu_inv(dm_act, dm_act, nb_steps);
    mat qx(dm_state, nb_steps);
    mat qu(dm_act, nb_steps);
    vec q0(nb_steps);
    vec q0_common(nb_steps);
    vec q0_softmax(nb_steps);

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);
    vec v0_softmax(nb_steps + 1);

    cube K(dm_act, dm_state, nb_steps);
    mat kff(dm_act, nb_steps);
    cube sigma_ctl(dm_act, dm_act, nb_steps);
    cube prec_ctl(dm_act, dm_act, nb_steps);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = Cxx.slice(nb_steps);
    v.col(nb_steps) = cx.col(nb_steps);
    v0(nb_steps) = c0(nb_steps);
    v0_softmax(nb_steps) = c0(nb_steps);

	for(int i = nb_steps - 1; i>= 0; --i)
	{
        // extra terms due to parameter distribution
         for (int j = 0; j < dm_state + dm_act + 1; j++){
            for (int k = 0; k < dm_state + dm_act + 1; k++){
                P(j, k) = trace(V.slice(i+1)*sigma_param.slice(i).submat(dm_state*j, dm_state*k, (j + 1)*dm_state - 1, (k + 1)*dm_state - 1));
        }}

        Pxx = P.submat(0, 0, dm_state - 1, dm_state - 1);
        Pxu = 2*P.submat(0, dm_state, dm_state - 1, dm_state + dm_act - 1);
        Puu = P.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1);
        px = 2*P.submat(0, dm_state + dm_act, dm_state - 1, dm_state + dm_act);
        pu = 2*P.submat(dm_state, dm_state + dm_act, dm_state + dm_act - 1, dm_state + dm_act);
        p0 = P(dm_state + dm_act, dm_state + dm_act);
    
        Qxx.slice(i) = (Cxx.slice(i) + A.slice(i).t() * V.slice(i+1) * A.slice(i) + Pxx) / alpha;
        Quu.slice(i) = (Cuu.slice(i) + B.slice(i).t() * V.slice(i+1) * B.slice(i) + Puu) / alpha;
        Qux.slice(i) = (Cxu.slice(i) + A.slice(i).t() * V.slice(i+1) * B.slice(i) + Pxu).t() / alpha;

        qu.col(i) = (cu.col(i) + 2.0 * B.slice(i).t() * V.slice(i+1) * c.col(i) + B.slice(i).t() * v.col(i+1) + pu) / alpha;
        qx.col(i) = (cx.col(i) + 2.0 * A.slice(i).t() * V.slice(i+1) * c.col(i) + A.slice(i).t() * v.col(i+1) + px) / alpha;
        q0_common(i) = as_scalar(c0(i) +  c.col(i).t() * V.slice(i+1) * c.col(i) + p0 + v.col(i+1).t() * c.col(i));

        q0(i) = (q0_common(i) + v0(i+1)) / alpha;
        q0_softmax(i) = (q0_common(i) + v0_softmax(i+1)) / alpha;

        if ((Quu.slice(i)).is_sympd()) {
            _diverge = i;
            break;
        }

        Quu_inv.slice(i) = inv(Quu.slice(i));
        K.slice(i) = - Quu_inv.slice(i) * Qux.slice(i);
        kff.col(i) = - 0.5 * Quu_inv.slice(i) * qu.col(i);

        sigma_ctl.slice(i) = - 0.5 * Quu_inv.slice(i);
        sigma_ctl.slice(i) = 0.5 * (sigma_ctl.slice(i).t() + sigma_ctl.slice(i));

        prec_ctl.slice(i) = - (Quu.slice(i).t() + Quu.slice(i));
        prec_ctl.slice(i) = 0.5 * (prec_ctl.slice(i).t() + prec_ctl.slice(i));

        V.slice(i) = (Qxx.slice(i) + Qux.slice(i).t() * K.slice(i)) * alpha;
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = (qx.col(i) + 2. * Qux.slice(i).t() * kff.col(i)) * alpha;
        v0(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i) - (0.5 * dm_act));
        v0_softmax(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0_softmax(i)
                         + 0.5 * (dm_act * log (2. * datum::pi) - log(det(- 2. * Quu.slice(i)))));
	}

    // transform outputs to numpy
    array_tf _Qxx = cube_to_array(Qxx);
    array_tf _Qux = cube_to_array(Qux);
    array_tf _Quu = cube_to_array(Quu);

    array_tf _qx = mat_to_array(qx);
    array_tf _qu = mat_to_array(qu);
    array_tf _q0 = mat_to_array(q0);
    array_tf _q0_softmax = mat_to_array(q0_softmax);

    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);
    array_tf _v0_softmax = vec_to_array(v0_softmax);

    array_tf _K = cube_to_array(K);
    array_tf _kff = mat_to_array(kff);
    array_tf _sigma_ctl = cube_to_array(sigma_ctl);

    py::tuple output =  py::make_tuple(_Qxx, _Qux, _Quu, _qx, _qu, _q0, _q0_softmax,
                                        _V, _v, _v0, _v0_softmax,
                                        _K, _kff, _sigma_ctl, _diverge);

    return output;
}



py::tuple parameter_backward_pass(array_tf _mu_xu, array_tf _sigma_xu,
                        array_tf _cx, array_tf _Cxx, array_tf _Cuu,
                        array_tf _cu, array_tf _Cxu, array_tf _c0,
                        array_tf _mu_param_nom, array_tf _sigma_param_nom,
                        array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                        double alpha, int dm_state, int dm_act, int nb_steps) {

    // inputs
    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    mat mu_xu = array_to_mat(_mu_xu);
    cube sigma_xu = array_to_cube(_sigma_xu);

    mat mu_param_nom = array_to_mat(_mu_param_nom);
    cube sigma_param_nom = array_to_cube(_sigma_param_nom);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    // temp
    mat mu_xu1 = join_vert(mu_xu, ones(1, nb_steps+1));
    cube sigma_xu1 = sigma_xu;
    sigma_xu1.insert_rows(dm_state + dm_act, 1);
    sigma_xu1.insert_cols(dm_state + dm_act, 1);
    
    int dm_param = dm_state*(dm_state + dm_act + 1);

    mat W(dm_param, dm_param);
    vec w(dm_param);

    mat P(dm_state + dm_act + 1, dm_state + dm_act + 1);
    mat Pxx(dm_state, dm_state);
    mat Pxu(dm_act, dm_state);
    mat Puu(dm_act, dm_act);
    vec px(dm_state);
    vec pu(dm_act);
    double p0;

    mat A_cl(dm_state, dm_state);
    vec c_cl(dm_state);    

    mat A_new(dm_state, dm_state);
    mat B_new(dm_state, dm_act);
    vec c_new(dm_state);

    // outputs
    mat mu_param(dm_param, nb_steps);
    cube sigma_param(dm_param, dm_param, nb_steps);

    cube prec_param_nom = sigma_param_nom;
    prec_param_nom.each_slice([](mat& X){ X = inv_sympd(X); } );

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = Cxx.slice(nb_steps);
    v.col(nb_steps) = cx.col(nb_steps);
    v0(nb_steps) = c0(nb_steps);

	for(int i = nb_steps - 1; i>= 0; --i)
	{
        // worst case parameter distribution
        W = kron(mu_xu1.col(i)*mu_xu1.col(i).t() + sigma_xu1.slice(i), V.slice(i+1));
        w = kron(mu_xu1.col(i), eye(dm_state, dm_state))*v.col(i+1);

        sigma_param.slice(i) = inv_sympd(prec_param_nom.slice(i) + 2/alpha*W);
        mu_param.col(i) = sigma_param.slice(i)*(prec_param_nom.slice(i)*mu_param_nom.col(i) - 1/alpha*w);

        // extra terms due to parameter distribution
         for (int j = 0; j < dm_state + dm_act + 1; j++){
            for (int k = 0; k < dm_state + dm_act + 1; k++){
                P(j, k) = trace(V.slice(i+1)*sigma_param.slice(i).submat(dm_state*j, dm_state*k, (j + 1)*dm_state - 1, (k + 1)*dm_state - 1));
        }};

        Pxx = Cxx.slice(i) + P.submat(0, 0, dm_state - 1, dm_state - 1);
        Pxu = Cxu.slice(i) + 2*P.submat(0, dm_state, dm_state - 1, dm_state + dm_act - 1);
        Puu = Cuu.slice(i) + P.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1);
        px = cx.col(i) + 2*P.submat(0, dm_state + dm_act, dm_state - 1, dm_state + dm_act);
        pu = cu.col(i) + 2*P.submat(dm_state, dm_state + dm_act, dm_state + dm_act - 1, dm_state + dm_act);
        p0 = c0(i) + P(dm_state + dm_act, dm_state + dm_act);

        A_new = reshape(mu_param(span(0, dm_state*dm_state - 1), i), size(A_new));
        B_new = reshape(mu_param(span(dm_state*dm_state, dm_state*(dm_state + dm_act) - 1), i), size(B_new));
        c_new = mu_param(span(dm_state*(dm_state + dm_act), dm_state*(dm_state + dm_act + 1 ) - 1), i);

        A_cl = A_new + B_new*K.slice(i);
        c_cl = B_new*kff.col(i) + c_new;
        
        V.slice(i) = A_cl.t()*V.slice(i+1)*A_cl + Pxx + Pxu*K.slice(i) + K.slice(i).t()*Puu*K.slice(i);
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = 2*A_cl*V.slice(i+1)*c_cl + Pxu*kff.col(i) + K.slice(i).t()*(pu + 2*Puu*kff.col(i)) + px + A_cl*v.col(i+1);

        v0(i) = v0(i+1) + p0 + as_scalar(c_cl.t()*V.slice(i+1)*c_cl + v.col(i+1).t()*c_cl + kff.col(i)*(Puu*kff.col(i) + pu))
                + trace((B_new.t()*V.slice(i+1)*B_new + Puu)*sigma_ctl.slice(i));
	}

    // transform outputs to numpy    
    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);
  
    array_tf _sigma_param = cube_to_array(sigma_param);
    array_tf _mu_param = mat_to_array(mu_param);

    py::tuple output =  py::make_tuple(_V, _v, _v0, _mu_param, _sigma_param,  _diverge);

    return output;
}

PYBIND11_MODULE(core, m)
{
    m.def("kl_divergence", &kl_divergence);
    m.def("quad_expectation", &quad_expectation);
    m.def("augment_cost", &augment_cost);
    m.def("forward_pass", &forward_pass);
    m.def("cubature_forward_pass", &cubature_forward_pass);
    m.def("backward_pass", &backward_pass);
    m.def("robust_backward_pass", &robust_backward_pass);
    m.def("parameter_backward_pass", &parameter_backward_pass);
}
