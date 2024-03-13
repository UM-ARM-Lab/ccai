import jax
import jax.numpy as jnp
import jax.experimental.checkify


class Adder:
    def __init__(self, a: jnp.array, b: jnp.array) -> None:
        self.a = a
        self.b = b

    def add(self, c: jnp.array) -> jnp.array:
        return self.a + self.b + c

    def compile(self):
        self.add = jax.jit(self.add).lower(jnp.ones((3,), dtype=jnp.int32)).compile()


if __name__ == "__main__":
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    c = jnp.array([7, 8, 9])
    adder = Adder(a, b)

    # adder.compile()
    # print(adder.add(c))

    lowered_add = jax.jit(adder.add).lower(jnp.ones((3,), dtype=jnp.int32))
    compiled_add = lowered_add.compile()
    print(compiled_add(c))
