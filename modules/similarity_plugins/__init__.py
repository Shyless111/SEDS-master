from modules.similarity_plugins.multilevel import MultiLevelSimilarityPlugin


def build_similarity_plugin(task_config):
    plugin_name = getattr(task_config, "similarity_plugin", "base")
    if plugin_name == "multilevel":
        return MultiLevelSimilarityPlugin(task_config)
    return None
