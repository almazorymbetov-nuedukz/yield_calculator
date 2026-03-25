====================================================================
BIODIESEL AND GLYCEROL EXCTRACTION CALCULATIONS
====================================================================
The core of the model is Multi-Layer Perceptron (MLP) with following functions:
1. Mish Activation Function - x*tanh(ln(1+e^x)): Better and smooth gradient movement, shows curvy relationships
2. Residual Blocks - x+f(x): Adds input back to the output of the layer
3. Layer Normalisation: Preventing high values to disrupt the learning process
4. Hansen Solubility Parameters (HSP): Calculates solubility distance and relative energy difference to predict thermodynamic relationships between DES and biodiesel
5. Arrhenius-Style Kinetic Mapping: Helps to model to clarify how fast molecular thickness and speed effects extraction process 
6. AdamW Optimizer: Prevents model overfitting and helps to understand general trends
7. Huber Loss: Prevents strange or extreme data to affect negatively on learning process
8. Monte Carlo Dropout: Doing about hundred calculations with different use of neural networks, and shows certainty of the calculation
9. Thermodynamic Saturation Index: If the solvent's volume is too small, uses limit-check in order to prevent high yield values


--------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------
1. You need Python.
2. Open terminal or command prompt.
3. Run the script by typing: python main.py
4. The script will automatically install any missing libraries.
5. The model will train itself (creating .pth and .joblib files).

--------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------
When you run the script, it will ask for 7 specific data points. Here 
is what they mean:

1. Temperature (K): The operating heat in Kelvin.
2. Molar Ratio: The chemical ratio of your DES (usually 2.0 or 4.0).
3. Density (g/cm3): How dense the mixture is.
4. Viscosity (mPa s): How thick the liquid is.
* DES/Oil Mass Ratio: How much solvent you are using compared to the oil.
* Water Content (%): Moisture in the batch.
* Initial Glycerol (%): The starting amount of the impurity.

--------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------
It will show three results:
1. Yield (%): How much of glycerol was successfully removed.
2. Residual Glycerol (%): The amount of impurity left in the fuel. 
3. Purity (%): Final biodiesel quality.

--------------------------------------------------------------------
NOTE
--------------------------------------------------------------------
In order to retrain the model, delete "model.pth", "sx.joblib" and "sy.joblib" files and run "main.py" again.