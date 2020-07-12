class BaseTextFeatureExtractor:
    def extract(self, corpus):
        """ Extract text feature for given corpus

        Inputs:
            corpus (lyricpsych.data.Corpus): data object

        Outputs:
            feature (lyricpsych.feature.TextFeature): output
        """
        raise NotImplementedError()
