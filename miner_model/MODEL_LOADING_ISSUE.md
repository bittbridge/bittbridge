# LSTM / Keras model loading issue on miner (Ubuntu VM)

This doc describes a model-loading error when running the miner on a Linux (Ubuntu) VM, how to reproduce it, and what was tried so far. The issue is not resolved yet

---

## The problem

When loading an LSTM model on the miner (on an Ubuntu VM), you get:

```text
ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}

TypeError: Error when deserializing class 'Dense' using config={...'quantization_config': None}.
Exception encountered: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

- **Faeze model** (created in notebook in early December): loads fine on the same VM
- **My model** (e.g. `lstm_model2.h5` / `lstm_model2.keras`): fails with the error above on the VM

So the same miner code and VM can load Faeze's model but not the one I save with the current setup

**Important:** Model loading works for both `.keras` and `.h5` when I do it **locally** with the simple workflow (`model.save(...)` and `load_model(...)`). The error appears only when I try to **load the model on the miner on the virtual machine**.

---

## How to reproduce

1. Use the Jupyter notebook with the **latest** TensorFlow (which installs the latest Keras)
2. Train and save the model (I tried both `.keras` and `.h5`)
3. Copy the saved model to the Ubuntu VM and run the miner (so the miner loads this model)
4. The miner fails with the `quantization_config` / Dense error above

---

## Context and assumption

My assumption is that the failure might be due to an **update in Keras or TensorFlow** after December. In early December (before update) I also tested saving LSTM and RNN models myself and they worked fine when loaded on the VM. Now, with the latest TensorFlow/Keras in the notebook, saving and then loading on the VM fails
**But it is just an assumption**

For now, the main issue is: when I use the Jupyter notebook with the latest TensorFlow (and thus the latest Keras), and save the model (I tried both `.keras` and `.h5`), loading that model on the miner on the Ubuntu VM produces this error

---

## What I tried

- Saving as **`.keras`** and as **`.h5`** — both fail when loaded on the miner on the VM (both work locally with `model.save` / `load_model`)
- Installing TensorFlow and then trying to **manually install an older Keras** (versions from before December). The issue is that TensorFlow automatically installs (or pulls in) the latest Keras, so pinning Keras alone did not help

## Possible direction to work in:
- One idea I have not tried yet: using **older TensorFlow** versions <2.20 so that the bundled Keras is also from that period - but I’m not sure if that’s the right direction
- Also, it could be the issue with update I made on 27th Jan  (check commits). We can try to restore version from Dec 12 -> it was working and was tested
- https://keras.io/guides/customizing_quantization/ -> maybe we should specify quantization config when save or load model? 

---

## References

- Keras saving: [Serialization and saving](https://keras.io/guides/serialization_and_saving/)
- In-repo: `miner_model/student_models/my_model.py` (model discovery and `load_model`)
