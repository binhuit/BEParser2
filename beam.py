from operator import itemgetter


class Beam:
    def __init__(self, beam_size=1):
        self.max_width = beam_size
        self.items = []

    def __iter__(self):
        for item in self.items:
            yield item

    def __len__(self):
        return len(self.items)

    def add(self, item, key='score'):
        if len(self) < self.max_width:
            self.items.append(item)
            self.items = sorted(self.items, key=itemgetter(key), reverse=True)
        else:
            if item[key] > self.items[-1][key]:
                self.items.append(item)
                self.items = sorted(self.items, key=itemgetter(key), reverse=True)
                self.items = self.items[:self.max_width]

    def top(self):
        try:
            return self.items[0]
        except Exception:
            raise


    def has_element(self, element):
        for item in self.items:
            if item == element:
                return True
        return False
