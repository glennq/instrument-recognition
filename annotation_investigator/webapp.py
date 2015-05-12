from flask import Flask, request, render_template, flash
from olform import *
import numpy as np
import pandas as pd
import cPickle
import os
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def inputData():
    global song_instr_map
    global grouping
    form = ReportForm(request.form)
    if request.method == 'POST' and form.validate():
        songs = form.songs.data
        group = int(form.group.data)
        instr_cnt = {}
        for i in songs:
            counted = set()
            for j in song_instr_map[i]:
                try:
                    instr = grouping[group][j] if j in grouping[group] else j
                except Exception, e:
                    print 'exception: {}'.format(e)
                if instr not in counted:
                    instr_cnt[instr] = instr_cnt.get(instr, 0) + 1
                counted.add(instr)
        del instr_cnt['S07']
        flash('Intrument counts: (in total {} instruments)'.
              format(len(instr_cnt)))
        for k, v in sorted(instr_cnt.items(), key=lambda x: x[1]):
            flash('{}:\t{}'.format(k, v))
    return render_template('form.html', form=form)

if __name__ == '__main__':
    with open('song_instr.pkl', 'rb') as f:
        song_instr_map = cPickle.load(f)
    groups = pd.read_csv(os.path.join(os.pardir, 'data', 'instGroup.csv'),
                         index_col=0)
    grouping = []
    for i in range(1, groups.shape[1]):
        grouping.append(dict(zip(groups['Instrument'].values,
                                 groups['Group {}'.format(i)].values)))
    app.secret_key = 'why would I tell you my secret k'
    app.run()
