import unittest
import torch
from functools import reduce
from open_dataset import create_class_folds, load_arxiv, load_plentoid, load_amazon_datasets, load_folds


dataset_names = ["cora", "citeseer", "photo", "computers", "ogb-arxiv"]

class Test_Dataset(unittest.TestCase):
    """Tests for GeekTechStuff Grafana API Python
    """
    

    def test_create_class_folds_non_overlapping(self):

        for name in dataset_names:
            
            if name == "ogb-arxiv":
                data = load_arxiv()
            elif (name == "cora") | (name == "citeseer"):
                data = load_plentoid(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
            elif name == "photo":
                data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
            elif name == "computers":
                data = load_amazon_datasets(name, train_portion=0.6, val_portion=0.2, test_portion=0.2, seed=0)
            else:
              raise NotImplementedError("Your choosen dataset: "+name+" is not supported")

    
            folds = create_class_folds(data, unknown_class_ratio=0.2)
            result = not reduce(set.intersection, map(set, folds))
            self.assertTrue(result)


    
    def test_train_test_sample_leak(self):

        for name in dataset_names:
            
            datasets = load_folds(name, unknown_class_ratio=0.4)
    
            for fold_id, data in enumerate(datasets):
                n_overlabp = torch.sum(data.labeled_mask & data.test_mask)
                self.assertEqual(int(n_overlabp), 0, "Sample overlap detected for: " + name + "on fold: "+str(fold_id))

    def test_train_test_class_leak(self):

        for name in dataset_names:
            
            datasets = load_folds(name, unknown_class_ratio=0.4)
    
            for fold_id, data in enumerate(datasets):
                train_classes = set(torch.unique(data.y[data.labeled_mask]).tolist())
                val_classes = set(torch.unique(data.y[data.unknown_class_val_mask]).tolist())
                test_classes = set(torch.unique(data.y[data.unknown_class_test_mask]).tolist())
                
                train_val = train_classes.intersection(val_classes)
                train_test = train_classes.intersection(test_classes)
                val_test = val_classes.intersection(test_classes)

                result = train_val.union(train_test, val_test)
                self.assertFalse(result, "Class overlap detected for: " + name + "on fold: "+str(fold_id))


if __name__ == '__main__':
    unittest.main()