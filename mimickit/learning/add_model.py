import learning.amp_model as amp_model

class ADDModel(amp_model.AMPModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return