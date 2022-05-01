import numpy as np

class EntityMetrics():

    def __init__(self, tag2id, id2tag):
        self.tag2id = tag2id
        self.id2tag = id2tag
        
    def report(self, targets, predicts):
        target_entities, predict_entities = self.extract_entities(targets, predicts)
        target_entities_by_name = self.category_by_name(target_entities)
        predict_entities_by_name = self.category_by_name(predict_entities)

        result = {name: dict() for name in target_entities_by_name}
        all_targets = set()
        all_predicts = set()

        supports = []
        precisions = []
        recalls = []
        f1s = []

        for name, _targets in target_entities_by_name.items():
            _targets = set(_targets)
            _predicts = set(predict_entities_by_name[name]) if name in predict_entities_by_name else set()

            true_positive =  len(_targets & _predicts)
            predict_cnt = len(_predicts)
            true_cnt = len(_targets)
            precision = true_positive / predict_cnt if predict_cnt > 0 else 0
            recall = true_positive / true_cnt if true_cnt > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            result[name]['support'] = true_cnt
            result[name]['precision'] = precision
            result[name]['recall'] = recall
            result[name]['f1'] = f1

            all_targets = all_targets.union(_targets)
            all_predicts = all_predicts.union(_predicts)
            supports.append(true_cnt)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        result['micro_avg'] = self.micro_avg(all_targets, all_predicts)
        result['macro_avg'] = {
            'support': np.sum(supports),
            'precision': np.average(precisions, weights=supports),
            'recall': np.average(recalls, weights=supports),
            'f1': np.average(f1s, weights=supports),
        }
        return result
    def print_result(self, result):
        names = [name for name in result if name not in set(['micro_avg','macro_avg'])]
        digits = 5
        width = max([len(name) for name in result])
        width = max(width, digits)
        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = u'\n'
        report += head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'
        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
        for _, name in enumerate(names):
            entity_result = result[name]
            report += row_fmt.format(*[name, entity_result['precision'], entity_result['recall'], entity_result['f1'], entity_result['support']], width=width, digits=digits)

        report += u'\n'
        micro_avg = result['micro_avg']
        report += row_fmt.format('micro avg',micro_avg['precision'], micro_avg['recall'], micro_avg['f1'], micro_avg['support'],width=width, digits=digits)
        macro_avg =  result['macro_avg']
        report += row_fmt.format('macro avg', macro_avg['precision'], macro_avg['recall'], macro_avg['f1'],
                                 macro_avg['support'], width=width, digits=digits)
        print(report)
    def micro_avg(self, targets, predicts):
        tp_cnt = len(targets & predicts)
        predict_cnt = len(predicts)
        true_cnt = len(targets)

        precision = tp_cnt / true_cnt if true_cnt > 0 else 0
        recall = tp_cnt / predict_cnt if predict_cnt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {
            'support': true_cnt,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def extract_entities(self, targets, predicts):

        pad_idx = self.tag2id['<pad>']
        target_tags = [np.array(tags) for tags in targets]
        predict_tags = [np.array(tags) for tags in predicts]
        mask = []
        for tags in target_tags:
            mask.append(tags != pad_idx)

        masked_targets = []
        masked_predicts = []
        for idx, row_mask in enumerate(mask):
            masked_targets.append(target_tags[idx][row_mask])
            masked_predicts.append(predict_tags[idx][row_mask])

        target_tags = masked_targets
        predict_tags = masked_predicts

        target_tags = self.id_to_tags(target_tags)
        predict_tags = self.id_to_tags(predict_tags)
        target_entities = [self.extract_entity_bio(tags) for tags in target_tags]
        predict_entities = [self.extract_entity_bio(tags) for tags in predict_tags]
        return target_entities, predict_entities
    def id_to_tags(self, tag_ids):
        return np.array([[self.id2tag[tag_id] for tag_id in tags] for tags in tag_ids])

    def extract_entity_bio(self, tags):
        entities = []  # [(entity, start, end)]
        entity = None
        last_idx = 0
        for idx, tag in enumerate(tags):
            if tag[0] == 'B':
                if entity is not None:
                    entities.append(entity)
                name = tag[2:]
                entity= [name, idx, idx]
            elif tag[0] == 'I':
                name = tag[2:]
                if entity is not None:
                    if entity[0] == name:
                        entity[2] = idx
                    else:
                        entity = None
            else:
                if entity is not None:
                    entities.append(entity)    
                entity = None
            last_idx = idx
        if entity is not None:
            entities.append(entity)
        return [tuple(entity) for entity in entities]
            
    def category_by_name(self, batch_entities):   # batch * entities
        nameToEntities = dict()
        for batch, entities in enumerate(batch_entities):
            for idx, entity in enumerate(entities):
                name = entity[0]
                if not name in nameToEntities:
                    nameToEntities[name] = []
                nameToEntities[name].append((batch, entity[1], entity[2]))  # (sentence_id, entity_start, entity_end)
        return nameToEntities
class Report:
    
    def __init__(self):
        pass
    
    def f1_score(self):
        pass