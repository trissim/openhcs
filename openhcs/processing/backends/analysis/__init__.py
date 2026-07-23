"""Image-analysis backend implementations.

Concrete backends are imported from their owning modules.  Keeping this package
initializer declaration-only prevents importing one backend from eagerly loading
every unrelated optional analysis runtime.
"""
