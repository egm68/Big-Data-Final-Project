# Visualizing and Understanding Dataset Search Results


Group Members: Sonia Castelo, Erin McGowan, and Shirley Berry


Course: Big Data CS-GY 6513 Section C

## Project structure

```
│
├── custom_functions.py      		  			<- Python functions used to ...
├── netgraph_functions.py      		  			<- Python functions used to ...
├── start_from_metadata.ipynb      		        <- See below
├── taxi_metadata_2023_05_04.csv                <- Taxi metadata generated from a search for the "taxi" keyword on May 5, 2023
├── DatasetsSummarizer_Tool_Demo.ipynb          <- Visualizations of taxi data in Jupyter notebook
├── requirements.txt/      		                <- Python package versions
├── start_from_metadata.ipynb/                  <- See below

```

## Notebooks
`full_pipeline.ipynb`: This notebook takes you through the entire pipeline: data ingestion, calculating dataset similarity, and visualizing the search results with the DatasetsSummarizer. This notebook will show you our results and walk you through the metadata pre-processing and similarity calculations in more detail. You can also use this notebook to change the search keyword and produce new results.
`start_from_metadata.ipynb`: This notebook takes you through the similarity calculations, starting from a metadata dataframe that has already been created. This notebook will show you our results, and walk you through the similarity calculations in more detail.
`DatasetsSummarizer_Tool_Demo.ipynb`: This notebook uses the DatasetsSummarizer library to create the summarizer visualization using pre-generated metadata.

## Reproducing
Figures 3, 4, 6, 7, and 8 as referenced in our paper can be reproduced and interacted with using the `DatasetsSummarizer_Tool_Demo.ipynb` notebook.