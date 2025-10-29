#!/usr/bin/env python
"""
Model Space Utilities
=====================

Utilities for converting between neural network model spaces and kinetic
model parameters. These functions enable integration between NN architectures
and kinetic modeling frameworks.

The module provides bidirectional conversion:
- Neural network layer specifications → kinetic rate parameters
- Kinetic model parameters → neural network model space format
"""

from typing import Dict


def convert_nn_rate_to_rate_dict(layer_attrs: dict) -> dict:
    """Convert neural network layer attributes to rate dictionary format.

    Parameters
    ----------
    layer_attrs : dict
        Dictionary containing layer attributes including SOURCE, TARGET,
        RANGE_ST, RANGE_D, and kernel_size

    Returns
    -------
    dict
        Rate dictionary with standardized keys: name, state_list,
        input_range, kernel_size, plus all original layer attributes
    """
    src_st, trg_st = layer_attrs["SOURCE"], layer_attrs["TARGET"]
    input_range = [
        layer_attrs["RANGE_ST"],
        layer_attrs["RANGE_ST"] + layer_attrs["RANGE_D"],
    ]
    ks = layer_attrs["kernel_size"]
    rate_name = "k_{}{}".format(src_st, trg_st)
    rate_dict = {
        "name": rate_name,
        "state_list": [src_st, trg_st],
        "input_range": input_range,
        "kernel_size": ks,
    }
    rate_dict.update(**layer_attrs)
    return rate_dict


def modelSpace_to_modelParams(model_arcs):
    """Convert a neural network 'model space' to kinetic model parameters.

    Transforms a neural network architecture representation into kinetic
    model parameters, creating states, rates, and scatter indices for
    TensorFlow operations.

    Parameters
    ----------
    model_arcs : list
        List of arc objects with Layer_attributes containing SOURCE, TARGET,
        and other layer specifications

    Returns
    -------
    dict
        Dictionary with keys:
        - States: sorted list of state identifiers
        - Rates: list of rate dictionaries with scatter_nd indices
        - Data: dict with contrib_rate_names list

    Example
    -------
    Expected YAML config structure::

        States: ['0', '1', '2', '3']
        Rates:
          - name: "k_{01}"
            state_list: ['0', '1']
            input_range: [5, 10]
          - name: "k_{10}"
            state_list: ['1', '0']
            input_range: [10, 15]
        Data:
          - contrib_rate_names: ['k_{30}']
    """
    kinetic_model_params = {
        "States": set([]),
        "Rates": [],
        "Data": {"contrib_rate_names": []},
    }
    states = sorted(
        set(
            [
                s
                for x in model_arcs
                for s in (x.Layer_attributes["SOURCE"], x.Layer_attributes["TARGET"])
            ]
        )
    )
    assert states
    # Create lookup table to place kinetic rates in a sparse matrix.
    # See comment below definition for 'scatter_nd' variable
    scatter_nd_lookup = {s: i for i, s in enumerate(states)}
    for arc in model_arcs:
        if not arc.Layer_attributes.get("EDGE", True):
            continue
        rate_dict = convert_nn_rate_to_rate_dict(arc.Layer_attributes)

        # Update the state list
        src_st, trg_st = rate_dict["state_list"]
        kinetic_model_params["States"].add(src_st)
        kinetic_model_params["States"].add(trg_st)

        # Scatter a flattened kinetic matrix to a sparse matrix as specified by indices.
        # scatter_nd: source is draining, source->target is increasing
        # For more on scatter_nd see https://www.tensorflow.org/api_docs/python/tf/scatter_nd
        scatter_nd = [
            ((scatter_nd_lookup[src_st], scatter_nd_lookup[src_st]), -1),
            ((scatter_nd_lookup[trg_st], scatter_nd_lookup[src_st]), +1),
        ]
        rate_dict["scatter_nd"] = scatter_nd

        # Update the rate list
        kinetic_model_params["Rates"] += [rate_dict]

        # Update the activity contribution list
        if arc.Layer_attributes.get("CONTRIB", False):
            kinetic_model_params["Data"]["contrib_rate_names"].append(rate_dict["name"])

    # Clean up state list
    kinetic_model_params["States"] = sorted(list(kinetic_model_params["States"]))
    return kinetic_model_params


def modelParams_to_modelSpace(model_params):
    """Convert kinetic model parameters back to neural network model space format.

    Inverse operation of modelSpace_to_modelParams, adding neural network
    specific fields like kernel_size and RANGE_* attributes.

    Parameters
    ----------
    model_params : dict
        Kinetic model parameters with States and Rates

    Returns
    -------
    dict
        Modified model_params with neural network model space fields added
    """
    scatter_nd_lookup = {s: i for i, s in enumerate(model_params["States"])}
    for rate in model_params["Rates"]:
        rate["kernel_size"] = 1
        rate["RANGE_ST"] = rate["input_range"][0]
        rate["RANGE_D"] = rate["input_range"][1] - rate["input_range"][0]
        s, t = rate["state_list"]
        # scatter_nd: source is draining, source->target is increasing
        scatter_nd = [
            ((scatter_nd_lookup[s], scatter_nd_lookup[s]), -1),
            ((scatter_nd_lookup[t], scatter_nd_lookup[s]), +1),
        ]
        rate["scatter_nd"] = scatter_nd
    return model_params
