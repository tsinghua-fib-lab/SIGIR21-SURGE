from visdom import Visdom
import numbers

class VizManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_vm'
        self.exp_name = flags_obj.name
        self.port = flags_obj.port
        self.set_visdom()

    def set_visdom(self):

        self.viz = Visdom(port=self.port, env=self.exp_name)

    def get_new_text_window(self, title):

        win = self.viz.text(title)

        return win

    def append_text(self, text, win):

        self.viz.text(text, win=win, append=True)

    def show_basic_info(self, flags_obj):

        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.learning_rate), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)

        self.basic = basic

        flags = self.viz.text('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            self.viz.text('{}: {}'.format(flag, value), win=flags, append=True)

        self.flags = flags

    def show_test_info(self):

        test = self.viz.text('Test Information:')
        self.test = test

    def step_update_line(self, title, value):

        if not isinstance(value, numbers.Number):
            return

        if not hasattr(self, title):

            setattr(self, title, self.viz.line([value], [0], opts=dict(title=title)))
            setattr(self, title + '_step', 1)

        else:

            step = getattr(self, title + '_step')
            self.viz.line([value], [step], win=getattr(self, title), update='append')
            setattr(self, title + '_step', step + 1)

    def step_update_multi_lines(self, kv_record):

        for title, value in kv_record.items():
            self.step_update_line(title, value)

    def plot_lines(self, y, x, opts):

        title = opts['title']
        if not hasattr(self, title):
            setattr(self, title, self.viz.line(y, x, opts=opts))
        else:
            self.viz.line(y, x, win=getattr(self, title), opts=opts, update='replace')

    def show_result(self, results):

        self.viz.text('-----Results-----', win=self.test, append=True)

        for metric, value in results.items():

            self.viz.text('{}: {}'.format(metric, value), win=self.test, append=True)

        self.viz.text('-----------------', win=self.test, append=True)