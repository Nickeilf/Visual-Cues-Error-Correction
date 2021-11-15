class Loss:
    """Accumulates and computes correctly training and validation losses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss = 0
        self._loss_task2 = 0
        self._denom = 0
        self._denom_task2 = 0
        self.batch_loss = 0
        self.batch_loss_task2 = 0

        self.loss_dict = {}

    def update(self, loss, n_items, task2=False):

        if not task2:
            # Store last batch loss
            self.batch_loss = loss.item()
            # Add it to cumulative loss
            self._loss += self.batch_loss
            # Normalize batch loss w.r.t n_items
            self.batch_loss /= n_items
            # Accumulate n_items inside the denominator
            self._denom += n_items
        else:
            self.batch_loss_task2 = loss.item()
            self._loss_task2 += self.batch_loss_task2
            self.batch_loss_task2 /= n_items
            self._denom_task2 += n_items

    def store_multi_loss(self, loss_dict):
        self.loss_dict['task'] = []
        self.loss_dict['loss'] = []
        for task, loss in loss_dict.items():
            self.loss_dict['task'].append(task)
            self.loss_dict['loss'].append(loss['loss'].item() / loss['n_items'])

    def get(self):
        if self._denom == 0:
            return 0
        if self._denom_task2 == 0:
            return self._loss / self._denom
        else:
            return self._loss / self._denom + self._loss_task2 / self._denom_task2

    def get_separate(self):
        if self._denom_task2 == 0:
            return self._loss / self._denom
        else:
            return self._loss / self._denom, self._loss_task2 / self._denom_task2

    @property
    def denom(self):
        return self._denom
