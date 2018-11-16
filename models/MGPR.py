import gpflow


class MGPR(gpflow.Parameterized):
    """
    We maintain multiple Gaussian Process for every dimension of the output.
    """
    def __init__(self, X0, Y0):
        super(MGPR, self).__init__()

        self.dim_outputs = Y0.shape[1]
        self.dim_input = X0.shape[1]

        self.models = []
        self.create_models(X0, Y0)

    def create_models(self, X0, Y0):
        for i in range(self.dim_outputs):
            kern = gpflow.kernels.RBF(input_dim=X0.shape[1], ARD=True)
            self.models.append(gpflow.models.GPR(X0, Y0[:, i:i+1], kern))
            self.models[i].clear()
            self.models[i].compile()

    def set_XY(self, X, Y):
        for i in range(len(self.models)):
            self.models[i].X = X
            self.models[i].Y = Y[:, i:i+1]

    def optimize(self):
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        for model in self.models:
            optimizer.minimize(model)






