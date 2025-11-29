# Street Safety in SD

This repository contains all of the code, final presentation PDF, and Jupyter notebook for the ECE 143 Group 3 (Evan, Jason, Ke, Raunak, and Zirui) final project.

## File Structure

**Important**: Within the `src` folder, there are subfolders containing the code written by each group member. Within these subfolders, there are **additional** `README` files further explaining the code within the respective subfolder.

Data and code is organized into the following folders:
- `data`: raw and processed data sets
    - `raw`: original data files (from city website)
    - `processed`: cleaned and merged data files (created by our code)
- `misc`: temporary data, any other files (if any)
- `src`: all python code for data processing, analysis, and visualization

The Jupyter notebook (`notebook.ipynb`) and final presentation PDF are located in the root folder.

## Running the Code

All code can be run from the Jupyter notebook in the `notebooks` folder. Before running the code, make sure to install all required third party modules listed in `requirements.txt`.

In the Jupyter notebook, simply run the cells in order to generate data (if needed) and display the visualizations. More information about can be found in the code and markdown cells within the notebook itself.

The intended way to explore the data and visualizations is through the Jupyter notebook.

Note: It may not be required to download, but there is one **missing** file that is too large to include in this repository: `roads_datasd.geojson`. This file can be downloaded from [here](https://seshat.datasd.org/gis_roads_all/roads_datasd.geojson) and should be placed in the `data/raw/roads_lines/` folder. Again, this file may not be required, but I wanted to include it here just in case.

## Required Third Party Modules
```
pandas
geopandas
folium
seaborn
numpy
matplotlib
ipykernel
```

These are also listed in `requirements.txt`.