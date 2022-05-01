import unittest
import numpy as np 
from finlp.metrics.entity import EntityMetrics

tags = ['O','B-LOC','I-LOC','B-ORG','I-ORG']

class EntityLevelMetricsTest(unittest.TestCase):
    def setUp(self):
        _tags = tags + ['<pad>']
        self.tag2id = {tag:idx for idx, tag in enumerate(_tags)}
        self.id2tag = {idx:tag for idx, tag in enumerate(_tags)}

    def test_id_to_tags(self):
        metrics = EntityMetrics(self.tag2id, self.id2tag)
        targets = [
            ['O','B-LOC','B-ORG','I-ORG','O','O','B-LOC','I-LOC','I-LOC','<pad>','<pad>'],
            ['O','O','B-ORG','I-ORG','O','O','B-LOC','I-LOC','O','O','B-LOC']
        ]
        tag_ids = [[self.tag2id[tag] for tag in tags] for tags in targets]
        converted = metrics.id_to_tags(tag_ids)
        self.assertEquals(targets, converted)

    def test_extract_entity_bio(self):
        metrics = EntityMetrics(self.tag2id, self.id2tag)
        tags = np.array(['O','B-LOC','B-ORG','I-ORG','O','O','B-LOC','I-LOC','I-LOC'])
        entities = metrics.extract_entity_bio(tags)
        self.assertEquals(3, len(entities))
    def test_category_by_name(self):
        entities = [
            [('LOC',3,5), ('ORG',8,10)],
            [('LOC', 3, 5)],
        ]
        metrics = EntityMetrics(self.tag2id, self.id2tag)
        nameToEnt = metrics.category_by_name(entities)
        self.assertEquals(2, len(nameToEnt['LOC']))
        self.assertEquals(1, len(nameToEnt['ORG']))
    def test_mask(self):
        rows = [[1,2,6,7,8],[2,6,8,5],[9,4,5]]
        rows = [np.array(row) for row in rows]
        mask = []
        for row in rows:
            mask.append(row != 5)
        masked_rows = []
        for idx, row_mask in enumerate(mask):
            masked_rows.append(rows[idx][row_mask])
        print(masked_rows)
    def test_metrics(self):
        metrics = EntityMetrics(self.tag2id, self.id2tag)

        targets = [
            ['O','B-LOC','B-ORG','I-ORG','O','O','B-LOC','I-LOC','I-LOC','<pad>','<pad>'],
            ['O','O','B-ORG','I-ORG','O','O','B-LOC','I-LOC','O','O','B-LOC'],
            ['O','B-LOC','B-ORG','I-ORG','O','O','B-LOC','I-LOC','O','O','O','B-LOC','<pad>'],
            ['B-LOC','I-LOC','O','O','O','O','B-LOC','I-LOC','O','O','B-ORG','I-ORG'],
        ]

        predicts = [
            ['O','B-LOC','B-ORG','I-ORG','O','O','B-LOC','I-LOC','I-LOC','<pad>','<pad>'],
            ['O','O','B-ORG','O','O','O','B-LOC','I-LOC','O','O','B-LOC'],
            ['O','B-LOC','B-ORG','I-ORG','O','O','O','I-LOC','O','O','O','B-LOC','<pad>'],
            ['B-LOC','I-LOC','O','O','O','O','B-LOC','I-LOC','O','O','B-ORG','I-ORG'],
        ]

        target_tag_ids = [[self.tag2id[tag] for tag in tags] for tags in targets]
        predict_tag_ids = [[self.tag2id[tag] for tag in tags] for tags in predicts]

        metrics.report(target_tag_ids, predict_tag_ids)