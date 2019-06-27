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


py::tuple backward_pass(array_tf _Q, array_tf _q,
                        array_tf _R, array_tf _r,
                        array_tf _P, array_tf _p,
                        array_tf _F, array_tf _G,
                        array_tf _T, array_tf _U,
                        array_tf _V, array_tf _X,
                        array_tf _Y, array_tf _Z,
                        double lmbda, int reg,
                        int nb_bdim, int nb_udim, int nb_steps) {

    // inputs
    cube Q = array_to_cube(_Q);
    mat q = array_to_mat(_q);

    cube R = array_to_cube(_R);
    mat r = array_to_mat(_r);

    cube P = array_to_cube(_P);
    mat p = array_to_mat(_p);

    cube F = array_to_cube(_F);
    cube G = array_to_cube(_G);

    cube T = array_to_cube(_T);
    cube U = array_to_cube(_U);
    cube V = array_to_cube(_V);

    cube X = array_to_cube(_X);
    cube Y = array_to_cube(_Y);
    cube Z = array_to_cube(_Z);

    // outputs
    cube C(nb_bdim, nb_bdim, nb_steps);
    mat c(nb_bdim, nb_steps);

    cube D(nb_udim, nb_udim, nb_steps);
    mat d(nb_udim, nb_steps);

    cube E(nb_udim, nb_bdim, nb_steps);
    mat e(nb_bdim * nb_bdim, nb_steps);

    cube Ereg(nb_udim, nb_bdim, nb_steps);
    cube Dreg(nb_udim, nb_udim, nb_steps);
    cube Dinv(nb_udim, nb_udim, nb_steps);

    cube S(nb_bdim, nb_bdim, nb_steps + 1);
    mat s(nb_bdim, nb_steps + 1);
    mat tau(nb_bdim * nb_bdim, nb_steps + 1);

    vec dS(2);

    cube Sreg(nb_bdim, nb_bdim, nb_steps + 1);

    cube K(nb_udim, nb_bdim, nb_steps);
    mat kff(nb_udim, nb_steps);

    int _diverge = 0;

    // init last time step
    S.slice(nb_steps) = Q.slice(nb_steps);
    s.col(nb_steps) = q.col(nb_steps);
    tau.col(nb_steps) = p.col(nb_steps);

	for(int i = nb_steps - 1; i>= 0; --i)
	{
        C.slice(i) = Q.slice(i) + F.slice(i).t() * S.slice(i+1) * F.slice(i);
        D.slice(i) = R.slice(i) + G.slice(i).t() * S.slice(i+1) * G.slice(i);
        E.slice(i) = (P.slice(i) + F.slice(i).t() * S.slice(i+1) * G.slice(i)).t();

        c.col(i) = q.col(i) + F.slice(i).t() * s.col(i+1) + T.slice(i).t() * tau.col(i+1)
                   + 0.5 * X.slice(i).t() * vectorise(S.slice(i+1));

        d.col(i) = r.col(i) + G.slice(i).t() * s.col(i+1) + V.slice(i).t() * tau.col(i+1)
                   + 0.5 * Z.slice(i).t() * vectorise(S.slice(i+1));

        e.col(i) = p.col(i) + U.slice(i).t() * tau.col(i) + 0.5 * Y.slice(i).t() * vectorise(S.slice(i+1));

        Sreg.slice(i+1) = S.slice(i+1);
        if (reg==2)
            Sreg.slice(i+1) += lmbda * eye(nb_bdim, nb_bdim);

        Ereg.slice(i) = (P.slice(i) + F.slice(i).t() * Sreg.slice(i+1) * G.slice(i)).t();

        Dreg.slice(i) = R.slice(i) + G.slice(i).t() * Sreg.slice(i+1) * G.slice(i);
        if (reg==1)
            Dreg.slice(i) += lmbda * eye(nb_udim, nb_udim);

        if (!(Dreg.slice(i)).is_sympd()) {
            _diverge = i;
            break;
        }

        Dinv.slice(i) = inv(Dreg.slice(i));
        K.slice(i) = - Dinv.slice(i) * Ereg.slice(i);
        kff.col(i) = - Dinv.slice(i) * d.col(i);

        dS += join_vert(kff.col(i).t() * d.col(i), 0.5 * kff.col(i).t() * D.slice(i) * kff.col(i));

        tau.col(i) = e.col(i);

        s.col(i) = c.col(i) + K.slice(i).t() * D.slice(i) * kff.col(i) +
                   K.slice(i).t() * d.col(i) + E.slice(i).t() * kff.col(i);

        S.slice(i) = C.slice(i) + K.slice(i).t() * D.slice(i) * K.slice(i) +
                     K.slice(i).t() * E.slice(i) + E.slice(i).t() * K.slice(i);
        S.slice(i) = 0.5 * (S.slice(i) + S.slice(i).t());
	}

    // transform outputs to numpy
    array_tf _S = cube_to_array(S);
    array_tf _s = mat_to_array(s);
    array_tf _tau = mat_to_array(tau);

    array_tf _dS = vec_to_array(dS);

    array_tf _K = cube_to_array(K);
    array_tf _kff = mat_to_array(kff);

    py::tuple output =  py::make_tuple(_S, _s, _tau,
                                       _dS, _K, _kff, _diverge);
	return output;
}


PYBIND11_MODULE(core, m)
{
    m.def("backward_pass", &backward_pass);
}
