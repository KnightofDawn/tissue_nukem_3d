import numpy as np
from copy import deepcopy

from vplants.meshing.implicit_surfaces import implicit_surface, implicit_surface_topomesh


class ParametricShapeModel:
    def __init__(self):
        self.parameters = {}
        self.parametric_function = None
        self.shape_model = {}
        self.density_function = None
        self.drawing_function = None

    def update_shape_model(self):
        self.shape_model = self.parametric_function(self.parameters)
        # print "Updated :",self.shape_model

    def perturbate_parameters(self,intensity,parameters_to_perturbate=None):
        if parameters_to_perturbate is None:
            parameters_to_perturbate = self.parameters.keys()
        for p in parameters_to_perturbate:
            self.parameters[p] += intensity - 2*intensity*np.random.rand()
        self.update_shape_model()

    def shape_model_density_function(self):
        return self.density_function(self.shape_model)

    def draw_shape_model(self,size,resolution):
        return self.drawing_function(self.shape_model,size,resolution)

    def parameter_optimization_annealing(self,energy,parameters_to_optimize=None,temperature=1):
        if parameters_to_optimize is None:
            parameters_to_optimize = self.parameters.keys()

        initial_energy = energy(self.parameters,self.shape_model_density_function())
        initial_energy += 20000.*temperature

        energy_variations = {}

        for p in self.parameters.keys():
            if p in parameters_to_optimize:
                deformed_model = deepcopy(self)
                deformed_model.parameters[p] += temperature
                deformed_model.update_shape_model()
                deformed_energy = energy(deformed_model.parameters,deformed_model.shape_model_density_function())
                energy_variations[p+"+"] = initial_energy-deformed_energy

                deformed_model = deepcopy(self)
                deformed_model.parameters[p] -= temperature
                deformed_model.update_shape_model()
                deformed_energy = energy(deformed_model.parameters,deformed_model.shape_model_density_function())
                energy_variations[p+"-"] = initial_energy-deformed_energy

        sorted_parameter_variations = np.array(energy_variations.keys())[np.argsort(-np.array(energy_variations.values()))]
        max_energy_variation = energy_variations[sorted_parameter_variations[0]]

        if max_energy_variation>0:
            # for p in sorted_parameter_variations[:1]:
            # for p in sorted_parameter_variations[:2]:
            for p in sorted_parameter_variations[:len(parameters_to_optimize)/2]:
            # for p in sorted_parameter_variations:
                if '+' in p:
                    self.parameters[p[:-1]] += temperature*energy_variations[p]/max_energy_variation
                elif '-' in p:
                    self.parameters[p[:-1]] -= temperature*energy_variations[p]/max_energy_variation

        self.update_shape_model()




