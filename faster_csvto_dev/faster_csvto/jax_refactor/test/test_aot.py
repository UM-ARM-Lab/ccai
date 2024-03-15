import jax
import jax.numpy as jnp
import jax.experimental.checkify
from functools import partial


class Adder:
    def __init__(self, a: jnp.array, b: jnp.array) -> None:
        self.a = a
        self.b = b
    #     self.add = self._add #jax.vmap(self._add)

    @staticmethod
    @partial(jax.vmap, in_axes=0, out_axes=0, static_argnums=0)
    def _add(self, c: jnp.array) -> jnp.array:
        return self.a + self.b + c

    # def __call__(self, c: jnp.array) -> jnp.array:
    #     self._add(c)

    # def compile(self):
    #     self.add = jax.jit(self.add).lower(jnp.ones((3,), dtype=jnp.int32)).compile()


if __name__ == "__main__":
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    c = jnp.array([7, 8, 9])
    d = jnp.array([a, b, c])
    adder = Adder(a, b)

    # adder.compile()
    # print(adder.add(c))

    # lowered_add = jax.jit(adder.add).lower(jnp.ones((3,), dtype=jnp.int32))
    # compiled_add = lowered_add.compile()
    # print(compiled_add(c))

    print(adder._add(c=d))
