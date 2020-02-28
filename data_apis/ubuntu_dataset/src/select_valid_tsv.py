import csv
import os
all_dialogs = os.getcwd() + '/dialogs'
os.chdir(all_dialogs)
list_dialogs = os.listdir(all_dialogs)

for i_dialogs in list_dialogs:
    ith_dialogs = os.getcwd() + "/" + i_dialogs
    os.chdir(ith_dialogs)
    ith_list = os.listdir(ith_dialogs)
    print("Dialogs Folder {}  # of tsv files: {}".format(i_dialogs, len(ith_list)))
    for m_tsv in ith_list:
        with open(m_tsv) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            name_col = []
            for row in reader:
                name_col.append(row[1])
            if len(set(name_col)) <= 2:
                print("Invalid tsv. Speakers < 3. Removing " + m_tsv)
                os.remove(m_tsv)
    os.chdir(all_dialogs)
