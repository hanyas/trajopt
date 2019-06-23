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


py::tuple backward_pass(array_tf _Rxx, array_tf _rx, array_tf _Ruu,
                        array_tf _ru, array_tf _Rxu,
                        array_tf _A, array_tf _B,
                        double lmbda, int reg,
                        int nb_xdim, int nb_udim, int nb_steps) {

    // inputs
    cube Rxx = array_to_cube(_Rxx);
    mat rx = array_to_mat(_rx);
    cube Ruu = array_to_cube(_Ruu);
    mat ru = array_to_mat(_ru);
    cube Rxu = array_to_cube(_Rxu);

    cube A = array_to_cube(_A);
    cube B = array_to_cube(_B);

    // outputs
    cube Q(nb_xdim + nb_udim, nb_xdim + nb_udim, nb_steps);
    cube Qxx(nb_xdim, nb_xdim, nb_steps);
    cube Qux(nb_udim, nb_xdim, nb_steps);
    cube Quu(nb_udim, nb_udim, nb_steps);
    mat qx(nb_xdim, nb_steps);
    mat qu(nb_udim, nb_steps);

    cube Qux_reg(nb_udim, nb_xdim, nb_steps);
    cube Quu_reg(nb_udim, nb_udim, nb_steps);
    cube Quu_inv(nb_udim, nb_udim, nb_steps);

    cube V(nb_xdim, nb_xdim, nb_steps + 1);
    mat v(nb_xdim, nb_steps + 1);
    vec dV(nb_xdim);

    cube V_reg(nb_xdim, nb_xdim, nb_steps + 1);

    cube K(nb_udim, nb_xdim, nb_steps);
    mat kff(nb_udim, nb_steps);

    int _diverge = 0;

	for(int i = nb_steps; i>= 0; --i)
	{
		if (i < nb_steps)
		{
			Qxx.slice(i) = Rxx.slice(i) + A.slice(i).t() * V.slice(i+1) * A.slice(i);
            Quu.slice(i) = Ruu.slice(i) + B.slice(i).t() * V.slice(i+1) * B.slice(i);
            Qux.slice(i) = (Rxu.slice(i) + A.slice(i).t() * V.slice(i+1) * B.slice(i)).t();

            qu.col(i) = ru.col(i) + B.slice(i).t() * v.col(i+1);
            qx.col(i) = rx.col(i) + A.slice(i).t() * v.col(i+1);

            V_reg.slice(i+1) = V.slice(i+1);
            if (reg==2)
                V_reg.slice(i+1) -= lmbda * eye(nb_xdim, nb_xdim);

            Qux_reg.slice(i) = (Rxu.slice(i) + A.slice(i).t() * V_reg.slice(i+1) * B.slice(i)).t();

            Quu_reg.slice(i) = Ruu.slice(i) + B.slice(i).t() * V_reg.slice(i+1) * B.slice(i);
            if (reg==1)
                Quu_reg.slice(i) -= lmbda * eye(nb_udim, nb_udim);

            if (!(-Quu_reg.slice(i)).is_sympd()) {
                _diverge = i;
                break;
            }

            Quu_inv.slice(i) = inv(Quu_reg.slice(i));
            K.slice(i) = - Quu_inv.slice(i) * Qux_reg.slice(i);
            kff.col(i) = - Quu_inv.slice(i) * qu.col(i);

            dV += join_vert(kff.col(i).t() * qu.col(i), 0.5 * kff.col(i).t() * Quu.slice(i) * kff.col(i));

            v.col(i) = qx.col(i) + K.slice(i).t() * Quu.slice(i) * kff.col(i) +
                       K.slice(i).t() * qu.col(i) + Qux.slice(i).t() * kff.col(i);

            V.slice(i) = Qxx.slice(i) + K.slice(i).t() * Quu.slice(i) * K.slice(i) +
                         K.slice(i).t() * Qux.slice(i) + Qux.slice(i).t() * K.slice(i);
            V.slice(i) = 0.5 * (V.slice(i) + V.slice(i).t());
		}
		else {
			V.slice(i) = Rxx.slice(i);
            v.col(i) = rx.col(i);
		}
	}

    // transform outputs to numpy
    array_tf _Qxx = cube_to_array(Qxx);
    array_tf _Qux = cube_to_array(Qux);
    array_tf _Quu = cube_to_array(Quu);

    array_tf _qx = mat_to_array(qx);
    array_tf _qu = mat_to_array(qu);

    array_tf _V = cube_to_array(V);
    array_tf _v = mat_to_array(v);
    array_tf _dV = vec_to_array(dV);

    array_tf _K = cube_to_array(K);
    array_tf _kff = mat_to_array(kff);

    py::tuple output =  py::make_tuple(_Qxx, _Qux, _Quu, _qx, _qu,
                                       _V, _v, _dV, _K, _kff, _diverge);
	return output;
}


PYBIND11_MODULE(core, m)
{
    m.def("backward_pass", &backward_pass);
}
