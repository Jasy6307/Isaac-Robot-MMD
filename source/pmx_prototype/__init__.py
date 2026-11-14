"""PMX runtime, training and offline tools with compatible module names."""
from .source_layout import SOURCE_DIR, MODULE_DIRS

# Preserve imports such as source.pmx_prototype.character_appearance.
__path__.extend(str(SOURCE_DIR / name) for name in MODULE_DIRS)
