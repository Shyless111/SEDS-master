from modules.similarity_plugins.multilevel import MultiLevelSimilarityPlugin
from modules.similarity_plugins.qamf import QueryAdaptiveLateFusionPlugin
from modules.similarity_plugins.adaptive_multilevel import AdaptiveMultiLevelSimilarityPlugin
from modules.similarity_plugins.difficulty_multilevel import DifficultyAwareMultiLevelSimilarityPlugin


def build_similarity_plugin(task_config, embed_dim=None):
    plugin_name = getattr(task_config, "similarity_plugin", "base")
    if plugin_name == "multilevel":
        return MultiLevelSimilarityPlugin(task_config)
    if plugin_name == "qamf":
        return QueryAdaptiveLateFusionPlugin(task_config)
    if plugin_name == "adaptive_multilevel":
        if embed_dim is None:
            raise ValueError("adaptive_multilevel requires the model embedding dimension")
        return AdaptiveMultiLevelSimilarityPlugin(task_config, embed_dim)
    if plugin_name == "difficulty_multilevel":
        if embed_dim is None:
            raise ValueError("difficulty_multilevel requires the model embedding dimension")
        return DifficultyAwareMultiLevelSimilarityPlugin(task_config, embed_dim)
    return None
