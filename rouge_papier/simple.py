import os
import pkg_resources
from subprocess import check_output, CalledProcessError
import pandas as pd
import re

from . import util


def to_dataframe(hypotheses, references, ngrams=None, length=None, 
                 length_unit="word"):

    AVG_R_PATT = r"X ROUGE-{} Average_R: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_P_PATT = r"X ROUGE-{} Average_P: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_F_PATT = r"X ROUGE-{} Average_F: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"

    rouge_path = pkg_resources.resource_filename(
        'rouge_papier', os.path.join('data', 'ROUGE-1.5.5.pl'))
    rouge_data_path = pkg_resources.resource_filename(
        'rouge_papier', 'rouge_data')

    #-n 4 -m -a -l 100 -x -c 95
    #-r 1000 -f A -p 0.5 -t 0
    #args = ["perl", rouge_path, "-H"]

    #print(check_output(args).decode("utf8"))

    args = ["perl", rouge_path, "-e", rouge_data_path, "-a"]

    if ngrams is not None:
        args.extend(["-n", str(ngrams)])

    if length is not None:
        if length_unit == "word":
            args.extend(["-l", str(length)])
        elif length_unit == "byte":
            args.extend(["-b", str(length)])
        else:
            raise Exception(
                ("length_unit must be either 'word' or 'byte' "
                 "but found {}").format(length_unit))


    with util.TempFileManager() as manager:
        input_paths = []
        for hyp, refs in zip(hypotheses, references):
            hyp_path = manager.create_temp_file(hyp)
            ref_paths = manager.create_temp_files(refs)
            input_paths.append([hyp_path, ref_paths])
        config_path = manager.create_temp_file(
            util.make_simple_config_text(input_paths))

        try:
            output = check_output(
                args + ["-z", "SPL", config_path]).decode("utf8")

        except CalledProcessError as e:
            print(e.output)
            raise e

        groups = []
        cols = []
        data = []
        if ngrams is not None:
            ngram_results = []
            for ngram in range(1, ngrams + 1):
                avg_r = float(
                    re.findall(AVG_R_PATT.format(ngram), output)[0][0])
                avg_p = float(
                    re.findall(AVG_P_PATT.format(ngram), output)[0][0])
                avg_f = float(
                    re.findall(AVG_F_PATT.format(ngram), output)[0][0])
                ngram_results.extend([avg_r, avg_p, avg_f])
                cols.extend(["Precision", "Recall", "F-Score"])
                groups.extend(["{}-gram".format(ngram)] * 3)
            data.append(ngram_results)
        return pd.DataFrame(data, index=["avg."],
            columns=pd.MultiIndex.from_tuples(zip(groups,cols)))
            
