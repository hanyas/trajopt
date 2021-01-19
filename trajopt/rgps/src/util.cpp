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


double kl_divergence(array_tf _p_K, array_tf _p_kff, array_tf _p_sigma_ctl,
                     array_tf _q_K, array_tf _q_kff, array_tf _q_sigma_ctl,
                     array_tf _mu_x, array_tf _sigma_x,
                     int dm_state, int dm_act, int nb_steps) {

    cube p_K = array_to_cube(_p_K);
    mat p_kff = array_to_mat(_p_kff);
    cube p_sigma_ctl = array_to_cube(_p_sigma_ctl);

    cube q_K = array_to_cube(_q_K);
    mat q_kff = array_to_mat(_q_kff);
    cube q_sigma_ctl = array_to_cube(_q_sigma_ctl);

    mat mu_x  = array_to_mat(_mu_x);
    cube sigma_x = array_to_cube(_sigma_x);

    double kl = 0.0;

    for(int i = 0; i < nb_steps; i++) {
        mat q_lambda_ctl = inv_sympd(q_sigma_ctl.slice(i));

        mat diff_K = (q_K.slice(i) - p_K.slice(i)).t() * q_lambda_ctl * (q_K.slice(i) - p_K.slice(i));
        mat diff_crs = (q_K.slice(i) - p_K.slice(i)).t() * q_lambda_ctl * (- q_kff.col(i) + p_kff.col(i));
        mat diff_kff = (- q_kff.col(i) + p_kff.col(i)).t() * q_lambda_ctl * (- q_kff.col(i) + p_kff.col(i));

        kl += as_scalar(0.5 * log( det(q_sigma_ctl.slice(i)) / det(p_sigma_ctl.slice(i)) )
		                + 0.5 * trace(q_lambda_ctl * p_sigma_ctl.slice(i))
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


py::tuple cubature_forward_pass(array_tf _mu_x0, array_tf _sigma_x0,
                                array_tf _mu_param, array_tf _sigma_param, array_tf _sigma_dyn,
                                array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                                int dm_state, int dm_act, int nb_steps) {

    // inputs
    vec mu_x0 = array_to_vec(_mu_x0);
    mat sigma_x0 = array_to_mat(_sigma_x0);

    mat mu_param = array_to_mat(_mu_param);
    cube sigma_param = array_to_cube(_sigma_param);
    cube sigma_dyn = array_to_cube(_sigma_dyn);

    mat A(dm_state, dm_state);
    mat B(dm_state, dm_act);
    vec c(dm_state);

    // cubature

    // particles for state, action, constant, ...
    // ... sigma_param, sigma_dyn
    int dm_augmented = dm_state + dm_act + 1 + dm_state + dm_state;
    vec mu_augmented(dm_augmented);
    mat chol_sigma_augmented(dm_augmented, dm_augmented);

    mat input_cubature_points(dm_augmented, 2 * dm_augmented);
    mat output_cubature_points(dm_state, 2 * dm_augmented);

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
        // get matrices from parameter distribution mean vector
        // reshape here is appled column-wise
        mat _A = mu_param(span(0, dm_state * dm_state - 1), i);
        mat _B = mu_param(span(dm_state * dm_state, dm_state * dm_state + dm_state * dm_act - 1), i);
        vec _c = mu_param(span(dm_state * dm_state + dm_state * dm_act, dm_state * dm_state + dm_state * dm_act + dm_state - 1), i);

        A = reshape(_A, size(A));
        B = reshape(_B, size(B));
        c = reshape(_c, size(c));

        // mu_u = K * mu_x + k
        mu_u.col(i) = K.slice(i) * mu_x.col(i) + kff.col(i);

        // sigma_u = sigma_ctl + K * sigma_x * K_T
        sigma_u.slice(i) = sigma_ctl.slice(i) + K.slice(i) * sigma_x.slice(i) * K.slice(i).t();
        sigma_u.slice(i) = 0.5 * (sigma_u.slice(i) + sigma_u.slice(i).t());

        // sigma_xu =   [[sigma_x,      sigma_x * K_T],
        //               [K * sigma_x,    sigma_u    ]]
        sigma_xu.slice(i) = join_vert(join_horiz(sigma_x.slice(i), sigma_x.slice(i) * K.slice(i).t()),
                                      join_horiz(K.slice(i) * sigma_x.slice(i), sigma_u.slice(i)));
        sigma_xu.slice(i) = 0.5 * (sigma_xu.slice(i) + sigma_xu.slice(i).t());

        // mu_xu =  [[mu_x],
        //           [mu_u]],
        mu_xu.col(i) = join_vert(mu_x.col(i), mu_u.col(i));

        // form augmented state mean and covariance
        mu_augmented = join_vert(mu_xu.col(i), ones(1), zeros<vec>(2 * dm_state));

        chol_sigma_augmented.fill(0.0);
        chol_sigma_augmented.submat(0, 0, dm_state + dm_act - 1, dm_state + dm_act - 1) = chol(symmatu(sigma_xu.slice(i)), "lower");
        chol_sigma_augmented.submat(dm_state + dm_act + 1, dm_state + dm_act + 1, dm_augmented - 1, dm_augmented - 1) = eye(2 * dm_state, 2 * dm_state);

        // calculate cubature points
        input_cubature_points = join_horiz(chol_sigma_augmented, - chol_sigma_augmented);
        input_cubature_points *= sqrt(dm_augmented);
        input_cubature_points.each_col() += mu_augmented;

        // propagate cubature points
         for (int j = 0; j < 2 * dm_augmented; j++){
            vec vec_xu = input_cubature_points.col(j);

            mat mat_xu = kron(vec_xu(span(0, dm_state + dm_act)).t(), eye(dm_state, dm_state));
            mat cov =  mat_xu * sigma_param.slice(i) * mat_xu.t();
            cov = 0.5 * (cov + cov.t());

            output_cubature_points.col(j) = join_horiz(join_horiz(A, B, c, chol(symmatu(cov), "lower")),
                                                        chol(symmatu(sigma_dyn.slice(i)), "lower")) * vec_xu;
        }

        // estimate new mean and covariance
        mu_x.col(i+1) = mean(output_cubature_points, 1);

        output_cubature_points.each_col() -= mu_x.col(i+1);

        sigma_x.slice(i+1).fill(0.0);
        output_cubature_points.each_col([&sigma_x, i] (vec& col){
            sigma_x.slice(i+1) += col * col.t();
        });
        sigma_x.slice(i+1) /= 2 * dm_augmented;
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


py::tuple policy_augment_cost(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
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
        mat lambda_ctl = inv_sympd(sigma_ctl.slice(i));

        agCxx.slice(i) = Cxx.slice(i) + 0.5 * alpha * K.slice(i).t() * lambda_ctl * K.slice(i);
        agCuu.slice(i) = Cuu.slice(i) + 0.5 * alpha * lambda_ctl;
        agCxu.slice(i) = Cxu.slice(i) - 0.5 * alpha * K.slice(i).t() * lambda_ctl;
        agcx.col(i) = cx.col(i) + alpha * K.slice(i).t() * lambda_ctl * kff.col(i);
        agcu.col(i) = cu.col(i) - alpha * lambda_ctl * kff.col(i);
        agc0(i) = as_scalar(c0(i) + 0.5 * alpha * log( det(2. * datum::pi * sigma_ctl.slice(i)) )
                            + 0.5 * alpha * kff.col(i).t() * lambda_ctl * kff.col(i));
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


py::tuple policy_backward_pass(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
                               array_tf _cu, array_tf _Cxu, array_tf _c0,
                               array_tf _mu_param, array_tf _sigma_param, array_tf _sigma_dyn,
                               double alpha, int dm_state, int dm_act, int nb_steps) {

    // inputs
    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    mat mu_param = array_to_mat(_mu_param);
    cube sigma_param = array_to_cube(_sigma_param);
    cube sigma_dyn = array_to_cube(_sigma_dyn);

    mat A(dm_state, dm_state);
    mat B(dm_state, dm_act);
    vec c(dm_state);

    mat P(dm_state + dm_act + 1, dm_state + dm_act + 1);
    mat Pxx(dm_state, dm_state);
    mat Pxu(dm_state, dm_act);
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

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);

    cube K(dm_act, dm_state, nb_steps);
    mat kff(dm_act, nb_steps);
    cube sigma_ctl(dm_act, dm_act, nb_steps);
    cube lambda_ctl(dm_act, dm_act, nb_steps);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = Cxx.slice(nb_steps);
    v.col(nb_steps) = cx.col(nb_steps);
    v0(nb_steps) = c0(nb_steps);

	for(int i = nb_steps - 1; i>= 0; --i)
	{
        // get matrices from parameter distribution mean vector
        // reshape here is appled column-wise
        mat _A = mu_param(span(0, dm_state * dm_state - 1), i);
        mat _B = mu_param(span(dm_state * dm_state, dm_state * dm_state + dm_state * dm_act - 1), i);
        vec _c = mu_param(span(dm_state * dm_state + dm_state * dm_act, dm_state * dm_state + dm_state * dm_act + dm_state - 1), i);

        A = reshape(_A, size(A));
        B = reshape(_B, size(B));
        c = reshape(_c, size(c));

        // extra terms due to parameter distribution
         for (int j = 0; j < dm_state + dm_act + 1; j++){
            for (int k = 0; k < dm_state + dm_act + 1; k++){
                P(j, k) = trace(sigma_param.slice(i).submat(j * dm_state, k * dm_state,
                                                            (j + 1) * dm_state - 1, (k + 1) * dm_state - 1) * V.slice(i+1));
            }
         }

        Pxx = P.submat(0, 0, dm_state - 1, dm_state - 1);
        Puu = P.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1);
        Pxu = P.submat(0, dm_state, dm_state - 1, dm_state + dm_act - 1);

        px = P.submat(0, dm_state + dm_act, dm_state - 1, dm_state + dm_act);
        pu = P.submat(dm_state, dm_state + dm_act, dm_state + dm_act - 1, dm_state + dm_act);
        p0 = P(dm_state + dm_act, dm_state + dm_act);

        Qxx.slice(i) = - (Cxx.slice(i) + A.t() * V.slice(i+1) * A + Pxx) / alpha;
        Quu.slice(i) = - (Cuu.slice(i) + B.t() * V.slice(i+1) * B + Puu) / alpha;
        Qux.slice(i) = - (Cxu.slice(i) + A.t() * V.slice(i+1) * B + Pxu).t() / alpha;

        qu.col(i) = - (cu.col(i) + 2.0 * B.t() * V.slice(i+1) * c + B.t() * v.col(i+1) + 2. * pu) / alpha;
        qx.col(i) = - (cx.col(i) + 2.0 * A.t() * V.slice(i+1) * c + A.t() * v.col(i+1) + 2. * px) / alpha;
        q0(i) = - as_scalar(c0(i) +  v0(i+1) + c.t() * V.slice(i+1) * c +
                            + trace(V.slice(i+1) * sigma_dyn.slice(i)) + v.col(i+1).t() * c + p0) / alpha;

        if ((Quu.slice(i)).is_sympd()) {
            _diverge = i;
            break;
        }

        Quu_inv.slice(i) = inv(Quu.slice(i));
        K.slice(i) = - Quu_inv.slice(i) * Qux.slice(i);
        kff.col(i) = - 0.5 * Quu_inv.slice(i) * qu.col(i);

        sigma_ctl.slice(i) = - 0.5 * Quu_inv.slice(i);
        sigma_ctl.slice(i) = 0.5 * (sigma_ctl.slice(i).t() + sigma_ctl.slice(i));

        lambda_ctl.slice(i) = - (Quu.slice(i).t() + Quu.slice(i));
        lambda_ctl.slice(i) = 0.5 * (lambda_ctl.slice(i).t() + lambda_ctl.slice(i));

        V.slice(i) = - alpha * (Qxx.slice(i) + Qux.slice(i).t() * K.slice(i));
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = - alpha * (qx.col(i) + 2. * Qux.slice(i).t() * kff.col(i));
        v0(i) = - alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i)
                           + 0.5 * (dm_act * log (2. * datum::pi) - log(det(- 2. * Quu.slice(i)))));
	}

    // transform outputs to numpy
    array_tf _Qxx = cube_to_array(Qxx);
    array_tf _Qux = cube_to_array(Qux);
    array_tf _Quu = cube_to_array(Quu);

    array_tf _qx = mat_to_array(qx);
    array_tf _qu = mat_to_array(qu);
    array_tf _q0 = mat_to_array(q0);

    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);

    array_tf _K = cube_to_array(K);
    array_tf _kff = mat_to_array(kff);
    array_tf _sigma_ctl = cube_to_array(sigma_ctl);

    py::tuple output =  py::make_tuple(_Qxx, _Qux, _Quu, _qx, _qu, _q0,
                                        _V, _v, _v0,
                                        _K, _kff, _sigma_ctl, _diverge);

    return output;
}


py::tuple parameter_augment_cost(array_tf _mu_nominal, array_tf _sigma_nominal,
                                 double beta, int dm_param, int nb_steps) {

    // inputs
    mat mu_nominal = array_to_mat(_mu_nominal);
    cube sigma_nominal = array_to_cube(_sigma_nominal);

    // outputs
    cube agCxx(dm_param, dm_param, nb_steps);
    mat agcx(dm_param, nb_steps);
    vec agc0(nb_steps);

    for (int i = 0; i < nb_steps; i++) {
        mat lambda_nominal = inv_sympd(sigma_nominal.slice(i));

        agCxx.slice(i) = 0.5 * beta * lambda_nominal;
        agcx.col(i) = - beta * lambda_nominal * mu_nominal.col(i);
        agc0(i) = as_scalar(0.5 * beta * log( det(2. * datum::pi * sigma_nominal.slice(i)) )
                            + 0.5 * beta * mu_nominal.col(i).t() * lambda_nominal * mu_nominal.col(i));
    }

    // transform outputs to numpy
    array_tf _agCxx = cube_to_array(agCxx);
    array_tf _agcx = mat_to_array(agcx);
    array_tf _agc0 = vec_to_array(agc0);

    py::tuple output =  py::make_tuple(_agCxx, _agcx, _agc0);
    return output;
}

py::tuple parameter_backward_pass(array_tf _mu_x, array_tf _sigma_x,
                                  array_tf _K, array_tf _kff, array_tf _sigma_ctl, array_tf _sigma_dyn,
                                  array_tf _cx, array_tf _Cxx, array_tf _Cuu,
                                  array_tf _cu, array_tf _Cxu, array_tf _c0,
                                  array_tf _agCpp, array_tf _agcp, array_tf _agc0,
                                  double beta, int dm_state, int dm_act, int dm_param, int nb_steps) {

    // inputs
    mat mu_x = array_to_mat(_mu_x);
    cube sigma_x = array_to_cube(_sigma_x);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    cube sigma_dyn = array_to_cube(_sigma_dyn);

    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    cube agCpp = array_to_cube(_agCpp);
    mat agcp = array_to_mat(_agcp);
    vec agc0 = array_to_vec(_agc0);

    // recreate state-action-offset dist.
    mat mu_u(dm_act, nb_steps);
    cube sigma_u(dm_act, dm_act, nb_steps);

    mat mu_xu(dm_state + dm_act + 1, nb_steps + 1);
    cube sigma_xu(dm_state + dm_act + 1, dm_state + dm_act + 1, nb_steps + 1);

    for (int i = 0; i < nb_steps; i++) {
        // mu_u = K * mu_x + k
        mu_u.col(i) = K.slice(i) * mu_x.col(i) + kff.col(i);

        // sigma_u = sigma_ctl + K * sigma_x * K_T
        sigma_u.slice(i) = sigma_ctl.slice(i) + K.slice(i) * sigma_x.slice(i) * K.slice(i).t();
        sigma_u.slice(i) = 0.5 * (sigma_u.slice(i) + sigma_u.slice(i).t());

        // sigma_xu =   [[sigma_x,        sigma_x * K_T,   0.],
        //               [K * sigma_x,    sigma_u,         0.],
        //               [0.              0.               0.]]
        sigma_xu.slice(i) = join_vert( join_horiz(sigma_x.slice(i), sigma_x.slice(i) * K.slice(i).t(), zeros(dm_state, 1)),
                                       join_horiz(K.slice(i) * sigma_x.slice(i), sigma_u.slice(i), zeros(dm_act, 1)),
                                       join_horiz(zeros(1, dm_state), zeros(1, dm_act), zeros(1, 1)) );
        sigma_xu.slice(i) = 0.5 * (sigma_xu.slice(i) + sigma_xu.slice(i).t());

        // mu_xu =  [[mu_x],
        //           [mu_u],
        //           [1],
        mu_xu.col(i) = join_vert(mu_x.col(i), mu_u.col(i), ones(1));
    }

    mu_xu.col(nb_steps) = join_vert(mu_x.col(nb_steps), zeros(1), ones(1));
    sigma_xu.slice(nb_steps).submat(0, 0, dm_state - 1, dm_state - 1) = sigma_x.slice(nb_steps);

    // temp
    mat W(dm_param, dm_param);
    vec w(dm_param);

    mat A(dm_state, dm_state);
    mat B(dm_state, dm_act);
    vec c(dm_state);

    mat A_cl(dm_state, dm_state);
    vec c_cl(dm_state);
    mat sigma_block(dm_state + dm_act + 1, dm_state + dm_act + 1);

    mat P(dm_state + dm_act + 1, dm_state + dm_act + 1);
    mat Pxx(dm_state, dm_state);
    mat Pxu(dm_state, dm_act);
    mat Puu(dm_act, dm_act);
    vec px(dm_state);
    vec pu(dm_act);
    double p0;

    // outputs
    mat mu_optimal(dm_param, nb_steps);
    cube sigma_optimal(dm_param, dm_param, nb_steps);

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = - Cxx.slice(nb_steps);
    v.col(nb_steps) = - cx.col(nb_steps);
    v0(nb_steps) = - c0(nb_steps);

	for(int i = nb_steps - 1; i >= 0; --i)
	{
	    mat mat_mu_xu = kron(mu_xu.col(i).t(), eye(dm_state, dm_state));

	    mat Vpp = mat_mu_xu.t() * V.slice(i + 1) * mat_mu_xu + kron(sigma_xu.slice(i), V.slice(i + 1));
	    vec vp =  mat_mu_xu.t() * v.col(i + 1);

        W = 2.0 * (agCpp.slice(i) + Vpp) / beta;
        W = 0.5 * (W.t() + W);

        w = - (agcp.col(i) + vp) / beta;

        try {
            sigma_optimal.slice(i) = inv_sympd(W);
        } catch ( const std::runtime_error ) {
            _diverge = i;
            break;
        }
        sigma_optimal.slice(i) = 0.5 * (sigma_optimal.slice(i).t() + sigma_optimal.slice(i));

        mu_optimal.col(i) = sigma_optimal.slice(i) * w;

        mat _A = mu_optimal(span(0, dm_state * dm_state - 1), i);
        mat _B = mu_optimal(span(dm_state * dm_state, dm_state * dm_state + dm_state * dm_act - 1), i);
        vec _c = mu_optimal(span(dm_state * dm_state + dm_state * dm_act, dm_state * dm_state + dm_state * dm_act + dm_state - 1), i);

        A = reshape(_A, size(A));
        B = reshape(_B, size(B));
        c = reshape(_c, size(c));

        // extra terms due to parameter distribution
         for (int j = 0; j < dm_state + dm_act + 1; j++){
            for (int k = 0; k < dm_state + dm_act + 1; k++){
                P(j, k) = trace(sigma_optimal.slice(i).submat(j * dm_state, k * dm_state,
                                                              (j + 1) * dm_state - 1, (k + 1) * dm_state - 1) * V.slice(i+1));
            }
         }

        Pxx = P.submat(0, 0, dm_state - 1, dm_state - 1);
        Puu = P.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1);
        Pxu = P.submat(0, dm_state, dm_state - 1, dm_state + dm_act - 1);

        px = P.submat(0, dm_state + dm_act, dm_state - 1, dm_state + dm_act);
        pu = P.submat(dm_state, dm_state + dm_act, dm_state + dm_act - 1, dm_state + dm_act);
        p0 = P(dm_state + dm_act, dm_state + dm_act);

        A_cl = A + B * K.slice(i);
        c_cl = c + B * kff.col(i);
        sigma_block.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1) = sigma_ctl.slice(i);

        V.slice(i) = (- Cxx.slice(i) + Pxx) + K.slice(i).t() * (- Cuu.slice(i) + Puu) * K.slice(i)
                      + A_cl.t() * V.slice(i + 1) * A_cl + 2. * (- Cxu.slice(i) + Pxu) * K.slice(i);
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = (- cx.col(i) + 2. * px) + 2. * K.slice(i).t() * (- Cuu.slice(i) + Puu) * kff.col(i)
                    + 2. * (- Cxu.slice(i) + Pxu) * kff.col(i) + K.slice(i).t() * (- cu.col(i) + 2. * pu)
                    + 2. * A_cl.t() * V.slice(i + 1) * c_cl + A_cl.t() * v.col(i + 1);

        v0(i) = as_scalar( (- c0(i) + p0) + kff.col(i).t() * (- Cuu.slice(i) + Puu) * kff.col(i) + kff.col(i).t() * (- cu.col(i) + 2. * pu)
                            - trace(Cuu.slice(i + 1) * sigma_ctl.slice(i)) + v0(i + 1) + trace(V.slice(i + 1) * sigma_dyn.slice(i))
                            + mu_optimal.col(i).t() * kron(sigma_block, V.slice(i + 1)) * mu_optimal.col(i) + trace(kron(sigma_block, V.slice(i + 1)) * sigma_optimal.slice(i))
                            + c_cl.t() * V.slice(i + 1) * c_cl + c_cl.t() * v.col(i + 1) );
	}

    // transform outputs to numpy
    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);

    array_tf _mu_optimal = mat_to_array(mu_optimal);
    array_tf _sigma_optimal = cube_to_array(sigma_optimal);

    py::tuple output =  py::make_tuple(_V, _v, _v0, _mu_optimal, _sigma_optimal, _diverge);

    return output;
}


py::tuple parameter_dual_regularization(array_tf _p_mu, array_tf _p_sigma,
                                        array_tf _q_mu, array_tf _q_sigma,
                                        double kappa, int dm_state, int nb_steps) {

    // inputs
    mat p_mu = array_to_mat(_p_mu);
    cube p_sigma = array_to_cube(_p_sigma);

    cube p_lambda = p_sigma;
    p_lambda.each_slice([](mat& X){ X = inv_sympd(X); } );

    mat q_mu = array_to_mat(_q_mu);
    cube q_sigma = array_to_cube(_q_sigma);

    cube q_lambda = q_sigma;
    q_lambda.each_slice([](mat& X){ X = inv_sympd(X); } );

    // outputs
    cube regCxx(dm_state, dm_state, nb_steps + 1);
    mat regcx(dm_state, nb_steps + 1);
    vec regc0(nb_steps + 1);

    for (int i = 0; i < nb_steps + 1; i++) {
        regCxx.slice(i) = 0.5 * kappa * (p_lambda.slice(i) - q_lambda.slice(i));
        regcx.col(i) = - kappa * (p_lambda.slice(i) * p_mu.col(i) - q_lambda.slice(i) * q_mu.col(i));
        regc0(i) = as_scalar(- kappa
                             + 0.5 * kappa * log( det(2. * datum::pi * p_sigma.slice(i)) )
                             + 0.5 * kappa * p_mu.col(i).t() * p_lambda.slice(i) * p_mu.col(i)
                             - 0.5 * kappa * log( det(2. * datum::pi * q_sigma.slice(i)) )
                             - 0.5 * kappa * q_mu.col(i).t() * q_lambda.slice(i) * q_mu.col(i));
    }

    // transform outputs to numpy
    array_tf _regCxx = cube_to_array(regCxx);
    array_tf _regcx = mat_to_array(regcx);
    array_tf _regc0 = vec_to_array(regc0);

    py::tuple output =  py::make_tuple(_regCxx, _regcx, _regc0);
    return output;

}

py::tuple regularized_parameter_backward_pass(array_tf _mu_xu, array_tf _sigma_xu,
                                              array_tf _cx, array_tf _Cxx, array_tf _Cuu,
                                              array_tf _cu, array_tf _Cxu, array_tf _c0,
                                              array_tf _agCpp, array_tf _agcp, array_tf _agc0,
                                              array_tf _K, array_tf _kff, array_tf _sigma_ctl, array_tf _sigma_dyn,
                                              array_tf _regCxx, array_tf _regcx, array_tf _regc0,
                                              double beta, int dm_state, int dm_act, int dm_param, int nb_steps) {

    // inputs

    // augment state-action with offset
    mat mu_xu = join_vert(array_to_mat(_mu_xu), ones(1, nb_steps + 1));
    cube sigma_xu = array_to_cube(_sigma_xu);
    sigma_xu.insert_rows(dm_state + dm_act, 1);
    sigma_xu.insert_cols(dm_state + dm_act, 1);

    cube Cxx = array_to_cube(_Cxx);
    mat cx = array_to_mat(_cx);
    cube Cuu = array_to_cube(_Cuu);
    mat cu = array_to_mat(_cu);
    cube Cxu = array_to_cube(_Cxu);
    vec c0 = array_to_vec(_c0);

    cube regCxx = array_to_cube(_regCxx);
    mat regcx = array_to_mat(_regcx);
    vec regc0 = array_to_vec(_regc0);

    cube agCpp = array_to_cube(_agCpp);
    mat agcp = array_to_mat(_agcp);
    vec agc0 = array_to_vec(_agc0);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    cube sigma_dyn = array_to_cube(_sigma_dyn);

    // temp
    mat W(dm_param, dm_param);
    vec w(dm_param);

    mat A(dm_state, dm_state);
    mat B(dm_state, dm_act);
    vec c(dm_state);

    mat A_cl(dm_state, dm_state);
    vec c_cl(dm_state);
    mat sigma_block(dm_state + dm_act + 1, dm_state + dm_act + 1);

    mat P(dm_state + dm_act + 1, dm_state + dm_act + 1);
    mat Pxx(dm_state, dm_state);
    mat Pxu(dm_state, dm_act);
    mat Puu(dm_act, dm_act);
    vec px(dm_state);
    vec pu(dm_act);
    double p0;

    // outputs
    mat mu_optimal(dm_param, nb_steps);
    cube sigma_optimal(dm_param, dm_param, nb_steps);

    cube V(dm_state, dm_state, nb_steps + 1);
    mat v(dm_state, nb_steps + 1);
    vec v0(nb_steps + 1);

    int _diverge = 0;

    // last time step
    V.slice(nb_steps) = - Cxx.slice(nb_steps) + regCxx.slice(nb_steps);
    v.col(nb_steps) = - cx.col(nb_steps) + regcx.col(nb_steps);
    v0(nb_steps) = - c0(nb_steps) + regc0(nb_steps);

	for(int i = nb_steps - 1; i >= 0; --i)
	{
	    mat mat_mu_xu = kron(mu_xu.col(i).t(), eye(dm_state, dm_state));

	    mat Vpp = mat_mu_xu.t() * V.slice(i + 1) * mat_mu_xu + kron(sigma_xu.slice(i), V.slice(i + 1));
	    vec vp =  mat_mu_xu.t() * v.col(i + 1);

        W = 2.0 * (agCpp.slice(i) + Vpp) / beta;
        W = 0.5 * (W.t() + W);

        w = - (agcp.col(i) + vp) / beta;

        try {
            sigma_optimal.slice(i) = inv_sympd(W);
        } catch (...) {
            _diverge = i;
            break;
        }
        sigma_optimal.slice(i) = 0.5 * (sigma_optimal.slice(i).t() + sigma_optimal.slice(i));

        mu_optimal.col(i) = sigma_optimal.slice(i) * w;

        mat _A = mu_optimal(span(0, dm_state * dm_state - 1), i);
        mat _B = mu_optimal(span(dm_state * dm_state, dm_state * dm_state + dm_state * dm_act - 1), i);
        vec _c = mu_optimal(span(dm_state * dm_state + dm_state * dm_act, dm_state * dm_state + dm_state * dm_act + dm_state - 1), i);

        A = reshape(_A, size(A));
        B = reshape(_B, size(B));
        c = reshape(_c, size(c));

        // extra terms due to parameter distribution
         for (int j = 0; j < dm_state + dm_act + 1; j++){
            for (int k = 0; k < dm_state + dm_act + 1; k++){
                P(j, k) = trace(sigma_optimal.slice(i).submat(j * dm_state, k * dm_state,
                                                              (j + 1) * dm_state - 1, (k + 1) * dm_state - 1) * V.slice(i+1));
            }
         }

        Pxx = P.submat(0, 0, dm_state - 1, dm_state - 1);
        Puu = P.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1);
        Pxu = P.submat(0, dm_state, dm_state - 1, dm_state + dm_act - 1);

        px = P.submat(0, dm_state + dm_act, dm_state - 1, dm_state + dm_act);
        pu = P.submat(dm_state, dm_state + dm_act, dm_state + dm_act - 1, dm_state + dm_act);
        p0 = P(dm_state + dm_act, dm_state + dm_act);

        A_cl = A + B * K.slice(i);
        c_cl = c + B * kff.col(i);
        sigma_block.submat(dm_state, dm_state, dm_state + dm_act - 1, dm_state + dm_act - 1) = sigma_ctl.slice(i);

        V.slice(i) = (- Cxx.slice(i) + regCxx.slice(i) + Pxx) + K.slice(i).t() * (- Cuu.slice(i) + Puu) * K.slice(i)
                      + A_cl.t() * V.slice(i + 1) * A_cl + 2. * (- Cxu.slice(i) + Pxu) * K.slice(i);
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = (- cx.col(i) + regcx.col(i) + 2. * px) + 2. * K.slice(i).t() * (- Cuu.slice(i) + Puu) * kff.col(i)
                    + 2. * (- Cxu.slice(i) + Pxu) * kff.col(i) + K.slice(i).t() * (- cu.col(i) + 2. * pu)
                    + 2. * A_cl.t() * V.slice(i + 1) * c_cl + A_cl.t() * v.col(i + 1);

        v0(i) = as_scalar( (- c0(i) + regc0(i) + p0) + kff.col(i).t() * (- Cuu.slice(i) + Puu) * kff.col(i) + kff.col(i).t() * (- cu.col(i) + 2. * pu)
                            - trace(Cuu.slice(i + 1) * sigma_ctl.slice(i)) + v0(i + 1) +  trace(V.slice(i + 1) * sigma_dyn.slice(i))
                            + mu_optimal.col(i).t() * kron(sigma_block, V.slice(i + 1)) * mu_optimal.col(i) + trace(kron(sigma_block, V.slice(i + 1)) * sigma_optimal.slice(i))
                            + c_cl.t() * V.slice(i + 1) * c_cl + c_cl.t() * v.col(i + 1) );
	}

    // transform outputs to numpy
    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _v0 = vec_to_array(v0);

    array_tf _mu_optimal = mat_to_array(mu_optimal);
    array_tf _sigma_optimal = cube_to_array(sigma_optimal);

    py::tuple output =  py::make_tuple(_V, _v, _v0, _mu_optimal, _sigma_optimal, _diverge);

    return output;
}

PYBIND11_MODULE(core, m)
{
    m.def("kl_divergence", &kl_divergence);
    m.def("quad_expectation", &quad_expectation);
    m.def("cubature_forward_pass", &cubature_forward_pass);
    m.def("policy_augment_cost", &policy_augment_cost);
    m.def("policy_backward_pass", &policy_backward_pass);
    m.def("parameter_augment_cost", &parameter_augment_cost);
    m.def("parameter_backward_pass", &parameter_backward_pass);
    m.def("parameter_dual_regularization", &parameter_dual_regularization);
    m.def("regularized_parameter_backward_pass", &regularized_parameter_backward_pass);
}
