from .base import BaseTextFeatureExtractor


class InventoryScoreExtractor(BaseTextFeatureExtractor):
    def __init__(self, inventory=None, gensim_w2v=None,
                 compute_similarity=False):
        """
        """
        super().__init__()
        
        # this can take minutes if the model if big one
        if gensim_w2v is None:
            self.w2v = None
        else:
            self.w2v = gensim.downloader.load(gensim_w2v)

        self.inventory = inventory
        self.compute_similarity = compute_similarity
        self.load_inventory(inventory)
        
    def load_inventory(self, inventory):
        """
        """
        if inventory is None:
            