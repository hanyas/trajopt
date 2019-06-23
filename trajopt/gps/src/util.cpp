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
                       int nb_xdim, int nb_udim, int nb_steps) {

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
		                - 0.5 * nb_udim
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

py::tuple augment_reward(array_tf _Rxx, array_tf _rx, array_tf _Ruu,
                         array_tf _ru, array_tf _Rxu, array_tf _r0,
                         array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                         double alpha, int nb_xdim, int nb_udim, int nb_steps) {

    // inputs
    cube Rxx = array_to_cube(_Rxx);
    mat rx = array_to_mat(_rx);
    cube Ruu = array_to_cube(_Ruu);
    mat ru = array_to_mat(_ru);
    cube Rxu = array_to_cube(_Rxu);
    vec r0 = array_to_vec(_r0);

    cube K = array_to_cube(_K);
    mat kff = array_to_mat(_kff);
    cube sigma_ctl = array_to_cube(_sigma_ctl);

    // outputs
    cube agRxx(nb_xdim, nb_xdim, nb_steps + 1);
    mat agrx(nb_xdim, nb_steps + 1);
    cube agRuu(nb_udim, nb_udim, nb_steps + 1);
    mat agru(nb_udim, nb_steps + 1);
    cube agRxu(nb_xdim, nb_udim, nb_steps + 1);
    vec agr0(nb_steps + 1);

    for (int i = 0; i <= nb_steps; i++) {

        if(i < nb_steps) {
            mat prec_ctl = inv_sympd(sigma_ctl.slice(i));

            agRxx.slice(i) = Rxx.slice(i) - 0.5 * alpha * K.slice(i).t() * prec_ctl * K.slice(i);
            agRuu.slice(i) = Ruu.slice(i) - 0.5 * alpha * prec_ctl;
            agRxu.slice(i) = Rxu.slice(i) + 0.5 * alpha * K.slice(i).t() * prec_ctl;
            agrx.col(i) = rx.col(i) - alpha * K.slice(i).t() * prec_ctl * kff.col(i);
            agru.col(i) = ru.col(i) + alpha * prec_ctl * kff.col(i);
            agr0(i) = as_scalar(r0(i) - 0.5 * alpha * log( det(2. * datum::pi * sigma_ctl.slice(i)) )
                       - 0.5 * alpha * kff.col(i).t() * prec_ctl * kff.col(i));
        }
        else {
            agRxx.slice(i) = Rxx.slice(i);
            agrx.col(i) = rx.col(i);
            agRuu.slice(i) = Ruu.slice(i);
            agru.col(i) = ru.col(i);
            agRxu.slice(i) = Rxu.slice(i);
            agr0(i) = r0(i);
        }
    }

    // transform outputs to numpy
    array_tf _agRxx = cube_to_array(agRxx);
    array_tf _agrx = mat_to_array(agrx);
    array_tf _agRuu =  cube_to_array(agRuu);
    array_tf _agru = mat_to_array(agru);
    array_tf _agRxu =  cube_to_array(agRxu);
    array_tf _agr0 = vec_to_array(agr0);

    py::tuple output =  py::make_tuple(_agRxx, _agrx, _agRuu, _agru, _agRxu, _agr0);
    return output;
}

py::tuple forward_pass(array_tf _mu_x0, array_tf _sigma_x0,
                       array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_dyn,
                       array_tf _K, array_tf _kff, array_tf _sigma_ctl,
                       int nb_xdim, int nb_udim, int nb_steps) {

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
    mat mu_x(nb_xdim, nb_steps + 1);
    cube sigma_x(nb_xdim, nb_xdim, nb_steps + 1);

    mat mu_u(nb_udim, nb_steps);
    cube sigma_u(nb_udim, nb_udim, nb_steps);

    mat mu_xu(nb_xdim + nb_udim, nb_steps + 1);
    cube sigma_xu(nb_xdim + nb_udim, nb_xdim + nb_udim, nb_steps + 1);

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
            mu_xu.col(i+1) = join_vert(mu_x.col(i+1), zeros<vec>(nb_udim));
            sigma_xu.slice(i+1).submat(0, 0, nb_xdim - 1, nb_xdim - 1) = sigma_x.slice(i+1);
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


py::tuple backward_pass(array_tf _Rxx, array_tf _rx, array_tf _Ruu,
                        array_tf _ru, array_tf _Rxu, array_tf _r0,
                        array_tf _A, array_tf _B, array_tf _c, array_tf _sigma_dyn,
                        double alpha, int nb_xdim, int nb_udim, int nb_steps) {


    // inputs
    cube Rxx = array_to_cube(_Rxx);
    mat rx = array_to_mat(_rx);
    cube Ruu = array_to_cube(_Ruu);
    mat ru = array_to_mat(_ru);
    cube Rxu = array_to_cube(_Rxu);
    vec r0 = array_to_vec(_r0);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);
    mat c = array_to_mat(_c);
    cube sigma_dyn = array_to_cube(_sigma_dyn);

    // outputs
    cube Q(nb_xdim + nb_udim, nb_xdim + nb_udim, nb_steps);
    cube Qxx(nb_xdim, nb_xdim, nb_steps);
    cube Qux(nb_udim, nb_xdim, nb_steps);
    cube Quu(nb_udim, nb_udim, nb_steps);
    cube Quu_inv(nb_udim, nb_udim, nb_steps);
    mat qx(nb_xdim, nb_steps);
    mat qu(nb_udim, nb_steps);
    vec q0(nb_steps);
    vec q0_common(nb_steps);
    vec q0_softmax(nb_steps);

    cube V(nb_xdim, nb_xdim, nb_steps + 1);
    mat v(nb_xdim, nb_steps + 1);
    vec v0(nb_steps + 1);
    vec v0_softmax(nb_steps + 1);

    cube K(nb_udim, nb_xdim, nb_steps);
    mat kff(nb_udim, nb_steps);
    cube sigma_ctl(nb_udim, nb_udim, nb_steps);
    cube prec_ctl(nb_udim, nb_udim, nb_steps);

	for(int i = nb_steps; i>= 0; --i)
	{
		if (i < nb_steps)
		{
			Qxx.slice(i) = (Rxx.slice(i) + A.slice(i).t() * V.slice(i+1) * A.slice(i)) / alpha;
            Quu.slice(i) = (Ruu.slice(i) + B.slice(i).t() * V.slice(i+1) * B.slice(i)) / alpha;
            Qux.slice(i) = (Rxu.slice(i) + A.slice(i).t() * V.slice(i+1) * B.slice(i)).t() / alpha;

            qu.col(i) = (ru.col(i) + 2.0 * B.slice(i).t() * V.slice(i+1) * c.col(i) + B.slice(i).t() * v.col(i+1)) / alpha;
            qx.col(i) = (rx.col(i) + 2.0 * A.slice(i).t() * V.slice(i+1) * c.col(i) + A.slice(i).t() * v.col(i+1)) / alpha;
            q0_common(i) = as_scalar(r0(i) +  c.col(i).t() * V.slice(i+1) * c.col(i)
                            + trace(V.slice(i+1) * sigma_dyn.slice(i)) + v.col(i+1).t() * c.col(i));

            q0(i) = (q0_common(i) + v0(i+1)) / alpha;
            q0_softmax(i) = (q0_common(i) + v0_softmax(i+1)) / alpha;

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
            v0(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0(i) - (0.5 * nb_udim));
            v0_softmax(i) = alpha * (as_scalar(0.5 * qu.col(i).t() * kff.col(i)) + q0_softmax(i)
                             + 0.5 * (nb_udim * log (2. * datum::pi) - log(det(- 2. * Quu.slice(i)))));

		}
		else {
			V.slice(i) = Rxx.slice(i);
            v.col(i) = rx.col(i);
            v0(i) = r0(i);
            v0_softmax(i) = r0(i);
		}
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
                                        _K, _kff, _sigma_ctl);
	return output;
}


PYBIND11_MODULE(core, m)
{
    m.def("kl_divergence", &kl_divergence);
    m.def("quad_expectation", &quad_expectation);
    m.def("augment_reward", &augment_reward);
    m.def("forward_pass", &forward_pass);
    m.def("backward_pass", &backward_pass);
}
