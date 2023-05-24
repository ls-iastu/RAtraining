import time
import numpy as np
import jax
import optax
import tensorcircuit as tc


def main(n, nlayers, Jz, decay_steps, maxiter):
    K = tc.set_backend("jax")
    tc.set_dtype("complex128")

    ii = tc.gates._ii_matrix
    zz = tc.gates._zz_matrix

    g = tc.templates.graphs.Line1D(n)
    ncircuits = 500
    heih = tc.quantum.heisenberg_hamiltonian(
        g, hzz=Jz, hyy=1.0, hxx=1.0, hx=0, hy=0, hz=0
    )

    def energy(params, structures, n, nlayers):
        def one_layer(state, others):
            params, structures = others
            l = 0
            c = tc.Circuit(n, inputs=state)

            for i in range(0, n, 1):
                c.rx(i, theta=params[3 * l, i])
            for i in range(0, n, 1):
                c.ry(i, theta=params[3 * l + 1, i])

            for i in range(0, n, 2):
                matrix = structures[l, i] * ii + (1.0 - structures[l, i]) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            for i in range(1, n, 2):
                matrix = structures[l, i] * ii + (1.0 - structures[l, i]) * (
                    K.cos(params[3 * l + 2, i]) * ii
                    + 1.0j * K.sin(params[3 * l + 2, i]) * zz
                )
                c.any(
                    i,
                    (i + 1) % n,
                    unitary=matrix,
                )

            s = c.state()
            return s, s

        params = K.cast(K.real(params), dtype="complex128")
        structures = (K.sign(structures) + 1) / 2  # 0 or 1
        structures = K.cast(structures, dtype="complex128")

        c = tc.Circuit(n)
        s = c.state()
        s, _ = jax.lax.scan(
            one_layer,
            s,
            (
                K.reshape(params, [nlayers, 3, n]),
                K.reshape(structures, [nlayers, 1, n]),
            ),
        )
        c = tc.Circuit(n, inputs=s)

        e = tc.templates.measurements.operator_expectation(c, heih)
        return K.real(e)

    def redefine_params(params, structures):
        for i in range(structures.shape[0]):
            for j in range(structures.shape[1]):
                for k in range(structures.shape[2]):
                    if structures[i, j, k] > 0:
                        params[i, 3 * j + 2, k] = 0.0
        return params

    vagf = K.jit(
        K.vvag(energy, argnums=0, vectorized_argnums=(0, 1)), static_argnums=(2, 3)
    )

    # structure < 0: activated
    # structure > 0: unactivated
    # Only 10% two-qubit gates are activated
    structures = tc.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[ncircuits, nlayers, n]),
        dtype="complex128",
    )
    structures -= 0.1 * K.ones([ncircuits, nlayers, n])

    # initial parameters
    # the initial parameters of unactivated two-qubit gates are set to zeros
    params = np.random.uniform(
        low=0.0, high=2 * np.pi, size=[ncircuits, 3 * nlayers, n]
    )
    params = redefine_params(params, structures)
    params = tc.array_to_tensor(
        params,
        dtype="float64",
    )

    lr_schedule = optax.exponential_decay(
        init_value=1e-2, transition_steps=decay_steps, decay_rate=0.9
    )
    opt = K.optimizer(optax.adam(learning_rate=lr_schedule))

    e_list = []
    var_list = []
    for i in range(maxiter):
        if i % int(0.8 * maxiter / 10) == 0 and i != 0:
            structures -= 0.1 * K.ones([ncircuits, nlayers, n])
        time0 = time.time()
        e, grads = vagf(params, structures, n, nlayers)
        time1 = time.time()
        params = opt.update(grads, params)
        print(time1 - time0)
        e_list.append(K.numpy(e))
        var = grads * grads
        var = var[var > 1e-8]
        var_list.append(K.numpy(K.mean(var)))
    return e_list, var_list


if __name__ == "__main__":
    # Define hyperparameters
    Jz = 1.0
    decay_steps = 100
    n = 12
    nlayers = 12
    maxiter = 5000

    e_list, var_list = main(n, nlayers, Jz, decay_steps, maxiter)

    np.savez(
        "data.npz",
        e=e_list,
        var=var_list,
    )
