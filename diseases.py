import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-chex")


print(model.pathologies)


