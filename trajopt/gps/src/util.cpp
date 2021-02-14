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


py::tuple kl_divergence(array_tf _p_K, array_tf _p_kff, array_tf _p_sigma_ctl,
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

    vec kl(nb_steps);

    for(int i = 0; i < nb_steps; i++) {
        mat q_lambda_ctl = inv_sympd(q_sigma_ctl.slice(i));

        mat diff_K = (q_K.slice(i) - p_K.slice(i)).t() * q_lambda_ctl * (q_K.slice(i) - p_K.slice(i));
        mat diff_crs = (q_K.slice(i) - p_K.slice(i)).t() * q_lambda_ctl * (- q_kff.col(i) + p_kff.col(i));
        mat diff_kff = (- q_kff.col(i) + p_kff.col(i)).t() * q_lambda_ctl * (- q_kff.col(i) + p_kff.col(i));

        kl(i) = as_scalar(0.5 * log( det(q_sigma_ctl.slice(i)) / det(p_sigma_ctl.slice(i)) )
		                  + 0.5 * trace(q_lambda_ctl * p_sigma_ctl.slice(i))
		                  - 0.5 * dm_act
		                  + 0.5 * trace(diff_K * sigma_x.slice(i))
		                  + 0.5 * mu_x.col(i).t() * diff_K * mu_x.col(i)
		                  - mu_x.col(i).t() * diff_crs
		                  + 0.5 * diff_kff);
    }

    array_tf _kl = vec_to_array(kl);

    py::tuple output =  py::make_tuple(_kl);
    return output;
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
                       array_tf _alpha, int dm_state, int dm_act, int nb_steps) {

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

    vec alpha = array_to_vec(_alpha);

    // outputs
    cube agCxx(dm_state, dm_state, nb_steps + 1);
    mat agcx(dm_state, nb_steps + 1);
    cube agCuu(dm_act, dm_act, nb_steps + 1);
    mat agcu(dm_act, nb_steps + 1);
    cube agCxu(dm_state, dm_act, nb_steps + 1);
    vec agc0(nb_steps + 1);

    for (int i = 0; i < nb_steps; i++) {
        mat lambda_ctl = inv_sympd(sigma_ctl.slice(i));

        agCxx.slice(i) = Cxx.slice(i) + 0.5 * alpha(i) * K.slice(i).t() * lambda_ctl * K.slice(i);
        agCuu.slice(i) = Cuu.slice(i) + 0.5 * alpha(i) * lambda_ctl;
        agCxu.slice(i) = Cxu.slice(i) - 0.5 * alpha(i) * K.slice(i).t() * lambda_ctl;
        agcx.col(i) = cx.col(i) + alpha(i) * K.slice(i).t() * lambda_ctl * kff.col(i);
        agcu.col(i) = cu.col(i) - alpha(i) * lambda_ctl * kff.col(i);
        agc0(i) = as_scalar(c0(i) + 0.5 * alpha(i) * log( det(2. * datum::pi * sigma_ctl.slice(i)) )
                            + 0.5 * alpha(i) * kff.col(i).t() * lambda_ctl * kff.col(i));
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


py::tuple backward_pass(array_tf _Cxx, array_tf _cx, array_tf _Cuu,
                        array_tf _cu, array_tf _Cxu, array_tf _c0,
                        array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_dyn,
                        array_tf _alpha, int dm_state, int dm_act, int nb_steps) {

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

    vec alpha = array_to_vec(_alpha);

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
        Qxx.slice(i) = - (Cxx.slice(i) + A.slice(i).t() * V.slice(i+1) * A.slice(i)) / alpha(i);
        Quu.slice(i) = - (Cuu.slice(i) + B.slice(i).t() * V.slice(i+1) * B.slice(i)) / alpha(i);
        Qux.slice(i) = - (Cxu.slice(i) + A.slice(i).t() * V.slice(i+1) * B.slice(i)).t() / alpha(i);

        qu.col(i) = - (cu.col(i) + 2.0 * B.slice(i).t() * V.slice(i+1) * c.col(i) + B.slice(i).t() * v.col(i+1)) / alpha(i);
        qx.col(i) = - (cx.col(i) + 2.0 * A.slice(i).t() * V.slice(i+1) * c.col(i) + A.slice(i).t() * v.col(i+1)) / alpha(i);
        q0(i) = - as_scalar(c0(i) + v0(i+1) + c.col(i).t() * V.slice(i+1) * c.col(i)
                            + trace(V.slice(i+1) * sigma_dyn.slice(i)) + v.col(i+1).t() * c.col(i)) / alpha(i);

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

        V.slice(i) = - alpha(i) * (Qxx.slice(i) + Qux.slice(i).t() * K.slice(i));
        V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());

        v.col(i) = - alpha(i) * (qx.col(i) + 2. * Qux.slice(i).t() * kff.col(i));
        v0(i) = - alpha(i) * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i) - (0.5 * dm_act));
        v0(i) = - alpha(i) * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i)
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


PYBIND11_MODULE(core, m)
{
    m.def("kl_divergence", &kl_divergence);
    m.def("quad_expectation", &quad_expectation);
    m.def("augment_cost", &augment_cost);
    m.def("forward_pass", &forward_pass);
    m.def("backward_pass", &backward_pass);
}
