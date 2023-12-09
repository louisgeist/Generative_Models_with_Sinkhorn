
import torch
from ignite.metrics import InceptionScore
from ignite.engine import Engine
from classifiers import CNN_classifier
from gan import Generator
from sinkhorn_generative_model import Model

def OwnInceptionScore(data_name, gen_model_name):
    """
    Adapt the InceptionScore to our data
    """
    nb_sample = 200
    gen_model = torch.load(f'trained_models/{data_name}/{gen_model_name}.pt')
    class_model = torch.load(f'trained_models/{data_name}/classifier.pt')
    class_model.eval()
    def evaluation_step(engine, z):
        with torch.no_grad():
            gen_model.eval()
            if gen_model_name.split('_')[0] == "sinkhorn":
                return gen_model.forward_batch()
            else:
                return gen_model.forward(z)
    z = torch.rand(nb_sample, 2)
    evaluator = Engine(evaluation_step)
    metric = InceptionScore(num_features=10, feature_extractor=class_model)
    metric.attach(evaluator, "is")
    state = evaluator.run([z])
    return state.metrics["is"]

