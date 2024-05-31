<!-- ABOUT THE PROJECT -->
## Name Of Project TBD


Torch implementation for the TBD.

* Link to paper here


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.



## Usage
1. Install [PyTorch v2.0](https://github.com/pytorch/pytorch), scikit-learn and git   
`sudo pip install torch scikit-learn`   
`sudo apt-get install git`
2. Clone the code to local.    
`git clone https://github.com/AvitalRose/thesis.git thesis`  # need to change name
3. Run experiment on musk1, musk2, fox, elephant & tiger. 
Can change parameters in param.json  
`python main.py --dataset`    
The model, dataset, kde data, loss data & parameter tracker are saved to "results/[dataset].pkl"
The loss plot graph, kde graph & parameter tracker graph are saved to "results/[name of graph] [dataset].png"
4. Run experiment on wiki. 
Change parameters in param.json 
* run using preprocessed data from wiki.csv  
`python main.py "wiki"`    
* run using unprocessed data from wiki.json. Can change the default parameters directly when calling 
data.wiki_dataset.WikiDataset.
`python main.py "wiki" --process_wiki "wiki.json"`    
This will create files "wiki.csv" and "data\category_counter.pkl", "data\label2index.pkl", the latter two contain 
information on the processed data.
* run using costume wikipedia dataset:
  - Install dependencies 
  - create dataset:
    - Multi-Label dataset creation:  
    `python data\wiki_scraper.py "single_file.txt"`    
    The "single_file.txt" should be the list of wanted wikipedia pages, in an identical format to the one in example.txt.
    - Single-Label dataset creation:  
    `python data\wiki_scraper.py "file1.txt" "file2.txt" "file3.txt"`
    Each file.txt should be the list of wanted wikipedia pages, in an identical format to the one in example.txt.  
    In the Single-Label datasets, all pages listed in the "file_name.txt" get the label ["file_name"].  
  - The pages will be stored in the new_wiki_data.json which has the following data structure:
  A dictionary where each entry is a web page, and is stored as a dictionary with the keys: 
  ['categories', 'sections']), where:
      - "categories" is the list of labels extracted from the categories of the page according to wikipedia.
      - "sections" is a dictionary of all the instances, the key is the title of the section and the values are the
      text under the section in wikipedia.
  - run using the new data, (which now requires preprocessing, can change the default parameters directly when calling 
data.wiki_dataset.WikiDataset.):  
  `python main.py "wiki" --process-wiki "new_wiki_data.json"`    

The model, dataset, kde data, loss data & parameter tracker are saved to "results/wiki.pkl"
The loss plot graph, kde graph & parameter tracker graph are saved to "results/[name of graph] wiki.png"


## Models
The Simple model:    
![](./dec_model.png "Simple Model")

The Graph model:    
![](./dec_model.png "Graph Model")

The GMM model:    
![](./idec_model.png 'GMM model')
