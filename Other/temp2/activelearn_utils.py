#!/usr/bin/env python
""" Python utilities for Active-learning classification work.


NOTE: To parse ASAS (or other project) .arff and
      UPDATE MySQL table with feature values, see:
      generate_features_for_several_surveys.py:parse_arff_and_update_feature_tables()
       - which is called at the bottom of that python file.

"""

import sys, os
import random
from numpy import loadtxt
import numpy
import pprint

class Database_Utils:
    """ Establish database connections, contains methods related to database tables.
    """
    def __init__(self, pars={}):
        self.pars = pars
        self.connect_to_db()


    def connect_to_db(self):
        try:
            import MySQLdb
        except:
            return # only get here on the citris33 cluster
        self.tcp_db = MySQLdb.connect(host=pars['tcp_hostname'], \
                                  user=pars['tcp_username'], \
                                  db=pars['tcp_database'],\
                                  port=pars['tcp_port'])
        self.tcp_cursor = self.tcp_db.cursor()

        self.tutor_db = MySQLdb.connect(host=pars['tutor_hostname'], \
                                  user=pars['tutor_username'], \
                                  db=pars['tutor_database'], \
                                  passwd=pars['tutor_password'], \
                                  port=pars['tutor_port'])
        self.tutor_cursor = self.tutor_db.cursor()


    def retrieve_tutor_class_ids(self):
        """ Query tutor for the tutor class_id for each class used in R classifier.

        NOTE: The debosscher R indexed class names are takern from Joey's
              utils_classify.R::class.debos function
        """
        debclass_tutorclass = { \
            "a. Mira":"Mira",
            "b. Semireg PV":"Semiregular Pulsating Variable",
            "c. RV Tauri":"RV Tauri",
            "d. Classical Cepheid":"Classical Cepheid",
            "e. Pop. II Cepheid":"Population II Cepheid",
            "f. Multi. Mode Cepheid":"Multiple Mode Cepheid",
            "g. RR Lyrae, FM":"RR Lyrae, Fundamental Mode",
            "h. RR Lyrae, FO":"RR Lyrae, First Overtone",
            "i. RR Lyrae, DM":"RR Lyrae, Double Mode",
            "j. Delta Scuti":"Delta Scuti",
            "k. Lambda Bootis":"Lambda Bootis Variable",
            "l. Beta Cephei":"Beta Cephei",
            "m. Slowly Puls. B":"Slowly Pulsating B-stars",
            "n. Gamma Doradus":"Gamma Doradus",
            "o. Pulsating Be":"Be Star",
            "p. Per. Var. SG":"Periodically variable supergiants",
            "q. Chem. Peculiar":"Chemically Peculiar Stars",
            "r. Wolf-Rayet":"Wolf-Rayet",
            "s. T Tauri":"T Tauri",
            "t. Herbig AE/BE":"Herbig AE/BE Star",
            "u. S Doradus":"S Doradus",
            "v. Ellipsoidal":"Ellipsoidal",
            "w. Beta Persei":"Algol (Beta Persei)", #"Beta Persei",
            "x. Beta Lyrae":"Beta Lyrae",
            "y. W Ursae Maj.":"W Ursae Majoris",
            'Algol (Beta Persei)':"Algol (Beta Persei)"}

        rclass_tutorid_lookup = {}
        for deb_class, tutor_class in debclass_tutorclass.iteritems():
            select_str = 'SELECT class_id FROM classes WHERE class_name="%s" AND class_is_public="Yes"' % (tutor_class)
            self.tutor_cursor.execute(select_str)
            results = self.tutor_cursor.fetchall()
            if len(results) == 0:
                raise "Error"
            class_id = results[0][0]
            rclass_tutorid_lookup[deb_class] = class_id

        # TODO: want to query all class_name, then make sure they are not in the existing rclass_tutorid_lookup.values()
        select_str = 'SELECT class_name, class_id FROM classes where class_is_public="Yes"'
        self.tutor_cursor.execute(select_str)
        results = self.tutor_cursor.fetchall()
        if len(results) == 0:
            raise "Error"
        for row in results:
            (class_name, class_id) = row
            if class_id in rclass_tutorid_lookup.values():
                continue # do not add classes which we've already addeded to rclass_tutorid_lookup dictionary from debclass_tutorclass{}
            ### Sanity checks:
            #if (class_name in rclass_tutorid_lookup.keys()):
            #    print class_name, class_id, "Already exists   rclass_tutorid_lookup:", rclass_tutorid_lookup[class_name]
            #if class_name in debclass_tutorclass.values():
            #    for a,b in debclass_tutorclass.iteritems():
            #        if b == class_name:
            #            print '....debclass_tutorclass[', a, ']=', b, 'row.class_id=', class_id, rclass_tutorid_lookup.has_key(class_name)
            #    continue
            rclass_tutorid_lookup[class_name] = class_id

        #blah = rclass_tutorid_lookup.keys()
        #blah.sort()
        #for a in blah:
        #    print a, ':::', rclass_tutorid_lookup[a]
        #import pdb; pdb.set_trace()
        #print

        return rclass_tutorid_lookup


    def create_tables(self):
        """ Create MySQL tables needed for Active-Learning work.

        NOTE: Do this one time only.

        See "CFTDI Active Learning architecture" google-document for further descriptions.
        
        """

        """
CREATE TABLE activelearn_leaderboard (
act_session             SMALLINT,
act_iter                SMALLINT,
user_id         INT UNSIGNED,
perc_sciclass         DOUBLE,
perc_match_group      DOUBLE,
perc_match_al         DOUBLE,
overall_match_group   DOUBLE,
overall_match_al      DOUBLE,
INDEX (act_session, act_iter, user_id));
        """
        
        """
CREATE TABLE activelearn_user_class_stats (
act_session             SMALLINT,
act_iter                SMALLINT,
user_id         INT UNSIGNED,
tutor_class_id  SMALLINT,
n_correct       INTEGER,
n_total         INTEGER,
INDEX (act_session, act_iter, user_id, tutor_class_id));    
        """


        create_str = """CREATE TABLE activelearn_user_class (
user_id         INT UNSIGNED,
source_id       INT,
dtime           DATETIME,
tutor_class_id  SMALLINT,
other_class_id  SMALLINT,
user_period     DOUBLE,
confidence      TINYINT,
user_comment    VARCHAR(512),
PRIMARY KEY (user_id, source_id, dtime),
INDEX (tutor_class_id),
INDEX (other_class_id))
SELECT user_id, source_id, dtime, tutor_class_id, other_class_id, user_period, NULL as confidence, user_comment FROM activelearn_user_class__backup
        """
        #self.tcp_cursor.execute(create_str)


        create_str = """CREATE TABLE activelearn_algo_class (
act_id          INT UNSIGNED,
source_id       INT,
rank            SMALLINT,
tutor_class_id  SMALLINT,
prob            FLOAT,
PRIMARY KEY (act_id, source_id, rank))
        """
        #self.tcp_cursor.execute(create_str)


        create_str = """CREATE TABLE activelearn_session (
act_id                  INT UNSIGNED AUTO_INCREMENT,
act_session             SMALLINT,
act_iter                SMALLINT,
dtime                   DATETIME,
n_user_classifs         INT,
error_rate              FLOAT,
comment                 VARCHAR(1024),
PRIMARY KEY (act_session, act_iter),
INDEX (dtime),
INDEX (act_id))
        """
        #self.tcp_cursor.execute(create_str)


        ### This is intended for primary lookup of src_ids for a given act_id.  
        create_str = """
        CREATE TABLE activelearn_actid_srcid (
act_id                  INT UNSIGNED,
source_id               INT,
PRIMARY KEY (act_id, source_id))
        """
        #self.tcp_cursor.execute(create_str)

        ### This is seperate from the activelearn_actid_srcid due to not wanting to block it during inserting of new importances:
        create_str = """
        CREATE TABLE activelearn_srcid_importance (
act_id                  INT UNSIGNED,
source_id               INT,
importance              FLOAT,
INDEX (act_id, importance))
        """
        #self.tcp_cursor.execute(create_str)


    def add_data_into_new_tables(self):
        """ Fill new tables with data from existing old ones.
        - This is nessicary due to the first iteration not having the full
          set of tables created.

        """
        select_str = """
        INSERT INTO activelearn_actid_srcid (act_id, source_id)
        SELECT act_id, source_id FROM activelearn_algo_class WHERE rank=0
        """

        self.tcp_cursor.execute(select_str)
        results = self.tcp_cursor.fetchall()
        if len(results) == 0:
            raise "ERROR"


def compare_userclassifs_files(pars={}):
    """ Compare the original pars['user_classifs_fpaths'] activelearn
    user summary files (for each iteration) with Joey's new algorithmically generated
    user classif summary files.
    """

    DatabaseUtils = Database_Utils(pars=pars)
    rclass_tutorid_lookup = DatabaseUtils.retrieve_tutor_class_ids()

    tutorid_rclass = {}  # invert the dictionary
    for k,v in rclass_tutorid_lookup.iteritems():
        tutorid_rclass[v] = k

    orig_final_classifs = {}
    algo_final_classifs = {}
    
    for i in range(1, len(pars['user_classifs_joey_algo_chosen'])+1):
        tup_list = loadtxt(pars['user_classifs_joey_algo_chosen'][i],
                                         dtype={'names': ('src_id', 'class_id'),
                                                'formats': ('i4', 'i4')},
                                         usecols=(0,1))
        algo_final_classifs[i] = {}
        for src_id, class_id in tup_list:
            algo_final_classifs[i][src_id] = class_id

        tup_list = loadtxt(pars['user_classifs_fpaths'][i],
                                         dtype={'names': ('src_id', 'class_id'),
                                                'formats': ('i4', 'i4')},
                                         usecols=(0,1))
        orig_final_classifs[i] = {}
        for src_id, class_id in tup_list:
            orig_final_classifs[i][src_id] = class_id


    ### Now for each iteration, look up which sources match, etc
    for i in range(1, len(pars['user_classifs_joey_algo_chosen'])+1):
        print "i   src   match %20s %20s" % ("Original", "New algorithm")
        for orig_srcid, orig_classid in orig_final_classifs[i].iteritems():
            if algo_final_classifs[i].has_key(orig_srcid):
                match = (orig_classid == algo_final_classifs[i][orig_srcid])
                print "%d %d %6s %20s %20s" % (i, orig_srcid, str(match), tutorid_rclass[orig_classid], tutorid_rclass[algo_final_classifs[i][orig_srcid]])
            else:
                print "%d %d %6s %20s %20s" % (i, orig_srcid, '', tutorid_rclass[orig_classid], '')



    import pdb; pdb.set_trace()
    print


def get_simbad_matched_sources_which_overlap_userAL_sources(pars):
    """
        ### Do a data-cleaning check of sources in AL_SIMBAD_confirmed.dat file which
        ###  may be be repeats in earlier AL_addToTrain_?.dat files.
    """
    user_al_srcid_dict = {}
    for iter_i, al_fpath in pars['user_classifs_fpaths'].iteritems():
        if iter_i == 10:
            ### This is the simbad_match file
            data = loadtxt(al_fpath,
                                           dtype={'names': ('src_id', 'class_id'),
                                            'formats': ('i4', 'i4')},
                                           usecols=(0,1),
                                           unpack=True)
            simbad_src_list = data['src_id']
        else:
            data = loadtxt(al_fpath,
                                           dtype={'names': ('src_id', 'class_id'),
                                                  'formats': ('i4', 'i4')},
                                           usecols=(0,1),
                                           unpack=True)
            user_al_srcid_dict[iter_i] = data['src_id']

    aluser_overlap_sources = []
    for i, al_srcids in user_al_srcid_dict.iteritems():
        for al_srcid in list(set(simbad_src_list) & set(al_srcids)):
            if not al_srcid in aluser_overlap_sources:
                aluser_overlap_sources.append(al_srcid)
    print aluser_overlap_sources
    # Want to remove these sources from the SIMBAD dataset
    import pdb; pdb.set_trace()
    print
    

class IPython_Task_Administrator:
    """ Send of Imputation tasks

    Adapted from generate_weka_classifiers.py:Parallel_Arff_Maker()

    """
    def __init__(self, pars={}):
        try:
            from IPython.kernel import client
        except:
            pass

        self.kernel_client = client

        self.pars = pars
        # TODO:             - initialize ipython modules
        self.mec = client.MultiEngineClient()
        #self.mec.reset(targets=self.mec.get_ids()) # Reset the namespaces of all engines
        self.tc = client.TaskClient()
	self.task_id_list = []

        #### 2011-01-21 added:
        self.mec.reset(targets=self.mec.get_ids())
        self.mec.clear_queue()
        self.mec.clear_pending_results()
        self.tc.task_controller.clear()


    def initialize_clients(self):
        """ Instantiate ipython1 clients, import all module dependencies.
        """
	#task_str = """cat = os.getpid()"""
	#taskid = self.tc.run(client.StringTask(task_str, pull="cat"))
	#time.sleep(2)
	#print self.tc.get_task_result(taskid, block=False).results

        # 20090815(before): a = arffify.Maker(search=[], skip_class=False, local_xmls=True, convert_class_abrvs_to_names=False, flag_retrieve_class_abrvs_from_TUTOR=True, dorun=False)
        import time

        exec_str = """
import sys, os
sys.path.append(os.environ.get('TCP_DIR') + '/Algorithms')
import rpy2_classifiers
from rpy2.robjects.packages import importr
from rpy2 import robjects
from numpy import array
import rpy2.robjects.numpy2ri
"""
        self.mec.execute(exec_str)
	time.sleep(2) # This may be needed.

	### testing:
	#task_str = """cat = os.getpid()"""
	#taskid = self.tc.run(client.StringTask(task_str, pull="cat"))
	#time.sleep(1)
	#print self.tc.get_task_result(taskid, block=False).results


class Active_Learn(Database_Utils):
    """
        1) train simple randomforest on deboss data
        2) apply randomforet to current asas arff file
        3) store generated classifications, probs in a table
           - these can then be parsed by PHP code for web visualization
        4) Use additional iteration(s) of user classifications in training-set + Deboss
        5) Do steps 3-5
    """

    def __init__(self, pars={}):
        self.pars = pars
        self.connect_to_db()


    def fill_user_classifs(self, iter=0, session_id=0):
        """
        """
        lines = open(self.pars['user_classifs_fpaths'][iter]).readlines()
        
        srcid_classid_tups = []
        for line in lines:
            e = line.split()
            srcid_classid_tups.append((int(e[0]), int(e[1])))

        self.user_classifs[session_id][iter] = {'src_id':[],
                                    'tutor_class_id':[],
                                    'srcid_class_dict':{}}
        for (srcid, class_id) in srcid_classid_tups:
            self.user_classifs[session_id][iter]['src_id'].append(srcid)
            self.user_classifs[session_id][iter]['tutor_class_id'].append(class_id)
            self.user_classifs[session_id][iter]['srcid_class_dict'][srcid] = class_id
    

    def retrieve_user_consensus_classifications(self, iteration_id=None, session_id=0):
        """

        """
        self.user_classifs = {} # [session_id][iter_id]
        self.user_classifs[session_id] = {}

        for i in range(1, iteration_id):
            self.fill_user_classifs(iter=i, session_id=session_id)


    def insert_tups_into_rdb(self, rclass_tutorid_lookup={}, tup_list=[], actlearn_tups=[],
                             act_session_id=0, act_iter_id=0,
                             n_user_classifs=0, error_rate=0., session_iter_comment=""):
        """ Generate insert tuple list which will be inserted into the tranx RDB.
        """
        insert_str = 'INSERT INTO activelearn_session (act_session, act_iter, dtime, n_user_classifs, error_rate, comment) VALUES (%d, %d, NOW(), %d, %lf, "%s") ON DUPLICATE KEY UPDATE dtime=VALUES(dtime),  n_user_classifs=VALUES(n_user_classifs),  error_rate=VALUES(error_rate),  comment=VALUES(comment)' % (act_session_id, act_iter_id, n_user_classifs, error_rate, session_iter_comment)
        self.tcp_cursor.execute(insert_str)
        #import pdb; pdb.set_trace()
        #print

        select_str = "SELECT act_id FROM activelearn_session WHERE act_session=%d AND act_iter=%d" % ( \
                                      act_session_id, act_iter_id)
        self.tcp_cursor.execute(select_str)
        results = self.tcp_cursor.fetchall()
        if len(results) == 0:
            raise "ERROR"

        act_id = results[0][0]

        ### Need to delete all cases for this act_id (session_id, iter_id), since we are replacing the values,
        #      and there may be different source_id, so ON DUPLICATE UPDATE doesn't help.
        delete_str = "DELETE FROM activelearn_algo_class WHERE act_id=%d" % (act_id)
        self.tcp_cursor.execute(delete_str)

        delete_str = "DELETE FROM activelearn_srcid_importance WHERE act_id=%d" % (act_id)
        self.tcp_cursor.execute(delete_str)

        delete_str = "DELETE FROM activelearn_actid_srcid WHERE act_id=%d" % (act_id)
        self.tcp_cursor.execute(delete_str)


        #import pdb; pdb.set_trace()
        #print
        ### insert the total test set (albeit subsampled from 50k to < 30k):
        insert_list = ["INSERT INTO activelearn_algo_class (act_id, source_id, rank, tutor_class_id, prob) VALUES "]
        #time_to_quit = False
        for tup in tup_list:
            (src_id, rank, prob, deb_class) = tup
            # # # I think this bug has been fixed with a more complete filling of: rclass_tutorid_lookup{}
            #if not rclass_tutorid_lookup.has_key(deb_class):
            #    print '!!!', src_id, rank, prob, deb_class
            #    time_to_quit = True
            #    continue
            # # #            
            insert_list.append("(%d, %d, %d, %d, %lf), " % ( \
                               act_id, src_id, rank, rclass_tutorid_lookup[deb_class], prob))

        ### debug / test:   want to catch when we have a class (from the user) in the classifier which is not a debosscher class
        #if time_to_quit:
        #    # NOTE: realy should get here since it is due to a bug in the definition of the possible classes list
        #    print 'quitting: want to catch when we have a class (from the user) in the classifier which is not a debosscher class'
        #    sys.exit()

        insert_str = ''.join(insert_list)[:-2] + " ON DUPLICATE KEY UPDATE tutor_class_id=VALUES(tutor_class_id), prob=VALUES(prob)"

        self.tcp_cursor.execute(insert_str)


        ### insert the active learned user sourceid set (~100), with importances:
        
        insert_list = ["INSERT INTO activelearn_srcid_importance (act_id, source_id, importance) VALUES "]
        for tup in actlearn_tups:
            (src_id, importance) = tup
            insert_list.append("(%d, %d, %lf), " % ( \
                               act_id, src_id, importance))

        insert_str = ''.join(insert_list)[:-2] #+ " ON DUPLICATE KEY UPDATE tutor_class_id=VALUES(tutor_class_id), prob=VALUES(prob)"
        self.tcp_cursor.execute(insert_str)


        insert_list = ["INSERT INTO activelearn_actid_srcid (act_id, source_id) VALUES "]
        for tup in actlearn_tups:
            (src_id, importance) = tup
            insert_list.append("(%d, %d), " % ( \
                               act_id, src_id))

        insert_str = ''.join(insert_list)[:-2] #+ " ON DUPLICATE KEY UPDATE tutor_class_id=VALUES(tutor_class_id), prob=VALUES(prob)"
        self.tcp_cursor.execute(insert_str)



    # NOTE: done only at the beginning / creation of a session:
    def generate_initial_classifications(self, train_arff_str='', test_arff_str=''):
        """
        Input: project_id, arff_fpath

        Much of this is adapted from rp2_classifiers.py::__main__()

        NOTE: done only at the beginning / creation of a session.

        """
        do_ignore_NA_features = False # This is a strange option to set True, which would essentially skip a featrue from being used.  The hardcoded-exclusion features are sdss and ws_ related.
        skip_missingval_lines = False # we skip sources which have missing values

        algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
        sys.path.append(algo_code_dirpath)
        import rpy2_classifiers
        rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)

        #train_arff_str = open(os.path.expandvars("/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-02-05_23:43:21.830763/source_feats.arff")).read()
        traindata_dict = rc.parse_full_arff(arff_str=train_arff_str, skip_missingval_lines=skip_missingval_lines)
        #import pdb; pdb.set_trace()
        classifier_dict = rc.train_randomforest(traindata_dict,
                                                do_ignore_NA_features=do_ignore_NA_features,
                                                mtry=15, ntrees=1000, nodesize=1)


        #test_arff_str = open(os.path.expandvars("/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-02-06_00:03:02.699641/source_feats.arff")).read()
        testdata_dict = rc.parse_full_arff(arff_str=test_arff_str, skip_missingval_lines=skip_missingval_lines)

        r_name='rf_clfr'
        classifier_dict = {'class_name':r_name}

        classif_results = rc.apply_randomforest(classifier_dict=classifier_dict,
                                                data_dict=testdata_dict,
                                                do_ignore_NA_features=do_ignore_NA_features,
                                                return_prediction_probs=True)
        
        rclass_tutorid_lookup = self.retrieve_tutor_class_ids()

        tup_list = classif_results['predictions']['tups']
        #  form:     [(236518, 0, 0.36299999999999999, 'b. Semireg PV'), ...

        self.insert_tups_into_rdb(rclass_tutorid_lookup=rclass_tutorid_lookup,
                                  tup_list=tup_list,
                                  act_session_id=0,
                                  act_iter_id=0,
                                  n_user_classifs=0,
                                  error_rate=1.0,  # initally this error rate is not useful since it is in relation with the ASAS classifications which are incomplete.
                                  session_iter_comment="Initial testing session and ASAS classifications using raw deboss classifier")

        # TODO: now RDB INSERT the  classif_results['predictions'] tuple list.


    def get_orig_arff_datadicts(self, orig_trainset_arff_fpath="",
                                      orig_testset_arff_fpath="",
                                skip_missingval_lines=True):
        """  Given the original test and trainingset arff fpaths, retrieve dictionaries
        which contain extracted features, sourceids, classes, arff_rows.
        """

        train_arff_str = open(os.path.expandvars(orig_trainset_arff_fpath)).read()
        traindata_dict = self.rc.parse_full_arff(arff_str=train_arff_str, skip_missingval_lines=skip_missingval_lines, fill_arff_rows=True)

        test_arff_str = open(os.path.expandvars(orig_testset_arff_fpath)).read()
        testdata_dict = self.rc.parse_full_arff(arff_str=test_arff_str, skip_missingval_lines=skip_missingval_lines, fill_arff_rows=True)

        return {'traindata_dict':traindata_dict,
                'testdata_dict':testdata_dict,
                'train_arff_str':train_arff_str,
                'test_arff_str':test_arff_str,}

                
    def analysis_plot_period_amp(self, test_arff_rows=[], train_arff_rows=[],
                                 session_id=0, iteration_id=0,
                                 n_test_subsample=0,
                                 new_testset_rows=[],
                                 test_srcid_list=[],
                                 select_rows=[]):
        """ Plot the active learn found sources' period vs amps

        NOTE: It appears that now that we include high-confidence sources to the "test-set"
             (testset acuatlly meaning the sourses which will be presented to the user for actlearn),
             these high-confidence sources will not be plotted since thew are not in the "select_rows" list.
        
        """
        import pylab
        import numpy
        i_amp = 1
        i_freq = 20 # 14 with no colors
        #i_freq = 14 # 14 with no colors

        test_amp_list = []
        test_per_list = []
        #test_srcid_list=[],
        #actlearn_test_subset_ids=[]):
        #new_testset_rows
        #import pdb; pdb.set_trace()
        #print


        for row_str in select_rows:
            elems = row_str.split(',')
            test_amp_list.append(float(elems[i_amp]))
            test_per_list.append(float(elems[i_freq]))
        
        train_amp_list = []
        train_per_list = []
        for row_str in train_arff_rows:
            elems = row_str.split(',')
            train_amp_list.append(float(elems[i_amp]))
            train_per_list.append(float(elems[i_freq]))

        #pylab.plot(numpy.log10(numpy.array(train_per_list)), numpy.log10(numpy.array(train_amp_list)), 'ro', ms=4)
        #pylab.plot(numpy.log10(numpy.array(test_per_list)), numpy.log10(numpy.array(test_amp_list)), 'bo', ms=4)
        #pylab.xlabel("Log Period(days)")
        #pylab.ylabel("Log Amplitude(mag)")

        fig = pylab.figure()
        ax = fig.add_subplot('111')
        pylab.plot(numpy.array(train_amp_list), numpy.array(train_per_list), 'ro', ms=3)
        pylab.plot(numpy.array(test_amp_list), numpy.array(test_per_list), 'bo', ms=5)
        ax.set_ylabel("Log Frequency(1/days)")
        ax.set_xlabel("Log Amplitude(mag)")
        ax.set_title("Session=%d  Iteration=%d  N test sources=%d  Sources for users=%d" % (session_id,
                                                                       iteration_id,
                                                                       n_test_subsample,
                                                                       len(select_rows)))

        ax.set_xscale('log')
        ax.set_yscale('log')

        try:
            pylab.savefig('/home/pteluser/scratch/active_learn/actlearn_freq_amp_sess%d_iter%d_ntest%d_nuser%d.ps' % ( \
                                                                       session_id,
                                                                       iteration_id,
                                                                       n_test_subsample,
                                                                       len(select_rows)))
        except:
            print "!!! unable to write .ps file"
            pass
        #pylab.show()
        #import pdb; pdb.set_trace()
        #print


    def reduce_testset_using_costs(self, datadicts={}):
        """ Using some form of cost for each source, reduce the number of potential
        sources in the test dataset.

        Initially removing sources with 1day period alias.

        NOTE: This code need to update / return the following, everything else is not used later on:
        datadicts['testdata_dict']['srcid_list']
                                  ['arff_rows']

        """

        srcids_to_skip = []

        # KLUDGE: there are some duplicate sources in the ASAS dataset, want to remove one of the duplicates.  This list was added after iteration2, to train iteration=3 (second many user iteration) and later iterations:
        srcids_to_skip.extend(self.pars['asas_duplicate_sources'])
        
        # NOTE: 20,25 min window seems to show a significant/prominant number of sources at the 1-day period
        period_half_window = 30 * 1./(24.*60.) # in days
        new_srcid_list = []
        new_arff_rows = []
        new_class_list = []
        new_featname_longfeatval_dict = {}
        for featname in datadicts['testdata_dict']['featname_longfeatval_dict']:
            new_featname_longfeatval_dict[featname] = []
        for i, arff_row in enumerate(datadicts['testdata_dict']['arff_rows']):
            src_id = datadicts['testdata_dict']['srcid_list'][i]
            period = 1. / datadicts['testdata_dict']['featname_longfeatval_dict']['freq1_harmonics_freq_0'][i]
            if (((period <= (1. + period_half_window)) and
                 (period >= (1 - period_half_window)))
                or (src_id in srcids_to_skip)):
                # These periods fall within +- 30 minutes of 1 day alias.  (within siderial day)
                #  or these sources are in the ASAS duplicate source list

                # The following allows some period~1day alias sources to be included in output sources/arff_rows
                # KLUDGE: in session=0, iteration=1 (first), we used one source (217628) which initially did not have a found period ~1 day, so it was selected, but on iter=2 we use a different ARFF file for ASAS sources, so this source now falls within the ~1 day range.  So we want to allow this source through since it is needed for calculating ActiveLearning Costs.  This will continue to be a problem as we update the Nat/LS period finding algorithms. #on iter=0,2 we used 216527, which needs to be included even though on iter3 the period is ~1day
                if src_id not in ['217628', '216527', '245529', '252070', '217855', '242999']:
                    
                    continue # we skip from adding this source / row to the dataset
            new_srcid_list.append(datadicts['testdata_dict']['srcid_list'][i])
            new_arff_rows.append(arff_row)
            new_class_list.append(datadicts['testdata_dict']['class_list'][i])
            # KLUDGY & SLOW:
            for featname in datadicts['testdata_dict']['featname_longfeatval_dict']:
                new_featname_longfeatval_dict[featname].append( \
                    datadicts['testdata_dict']['featname_longfeatval_dict'][featname][i])
        datadicts['testdata_dict']['arff_rows'] = new_arff_rows
        datadicts['testdata_dict']['srcid_list'] = new_srcid_list
        datadicts['testdata_dict']['class_list'] = new_class_list
        datadicts['testdata_dict']['featname_longfeatval_dict'] = new_featname_longfeatval_dict
        #return datadicts


    def all_same(self, items):
            return all(x == items[0] for x in items)


    def condense_user_classified_sources(self, user_classif_lists):
        """
        ## NOTE: for now I will assume that we choose sources which have been classified by
              specific hardcoded users (which have classified everything reliably).
           - I also require that the tutor_classification is not None.
        ## Eventually, we may want to determine how many total users, and then retquire that 
             > 70% of classifications be identical for a source, given that 100% is when all users classify

           - TODO may also want datetime condition, or other_class_id condition (no JUNK)

        """        

        if 0:
            both_user_match_dict = {
                'src_id':[],
                'tutor_class_id':[],
                'iter_id':[]
                }


            # For iteration_id=2, this only finds 39 of the 46 sources which Joey choses for iteration=1
            #     So I just use the source,class he has decided upon.
            #required_user_ids = [1, 2] # iteration=2
            required_user_ids = range(1,50) # skip henrik(0) # [1, 2, 3, 4, 7, 11] # iteration=3
            len_required_user_ids = len(required_user_ids)
            
            srcid_classif_counts = {}

            # First, we will later need a list of all source_ids looked at by users for activelearn iteration:
            all_srcids_for_actlearn_iter = []
            for i, src_id in enumerate(user_classif_lists['src_id']):
                if ((user_classif_lists['user_id'][i] in required_user_ids)):
                    if src_id not in all_srcids_for_actlearn_iter:
                        all_srcids_for_actlearn_iter.append(src_id)

            for i, src_id in enumerate(user_classif_lists['src_id']):
                
                if ((user_classif_lists['user_id'][i] in required_user_ids) and
                    (user_classif_lists['tutor_class_id'][i] != None)):
                    if not srcid_classif_counts.has_key(src_id):
                        srcid_classif_counts[src_id] = []
                    #srcid_classif_counts[src_id].append(user_classif_lists['tutor_class_id'][i])
                    srcid_classif_counts[src_id].append((user_classif_lists['user_id'][i],
                                                         user_classif_lists['dtime'][i],
                                                         user_classif_lists['tutor_class_id'][i]))

            #src_id, user_id, class_id
            # - if src_id, user_id already exist, dont add\\
            # [(srcid, datetime, class), ...
            # [srcid][(userid,datetime,class)
            srcid_each_user_makes_some_classification = []
            for src_id, classif_list in srcid_classif_counts.iteritems():
                #if ((len(classif_list) == len_required_user_ids) and
                #    (self.all_same(classif_list))):
                classif_list.sort(reverse=True) # latest datetime first in list
                userids = []
                classids = []
                for user_id, dtime, class_id in classif_list:
                    if user_id not in userids:
                        userids.append(user_id)
                        classids.append(class_id)
                if len(classids) == len_required_user_ids:
                    srcid_each_user_makes_some_classification.append(src_id)

                uniq_classes = list(set(classids))
                # now get the count of classifications for each class
                #  - if count is == to len_required_user_ids, then use that class.
                #  - eventually this may be something less than a 100% match

                # # # # # # # #
                # # # # # # # #
                # TODO: Joey actually creates the 46 source iter=1 dataset by requiring:
                #       both classifications to match, or if two different classifications,
                #       then randomly choose between them.  (this is fine since he finds the classes pretty ambiguous)
                # TODO: do this instead of requiring a complete match (below):
                # # # # # # # #
                # # # # # # # #
                
                for a_class in uniq_classes:
                    n_users = classids.count(a_class)
                    if n_users == len_required_user_ids:
                        
                        # KLUDGEY: find the classif with the most classifications.
                        
                        i = user_classif_lists['src_id'].index(src_id)
                        tutor_class_id = user_classif_lists['tutor_class_id'][i]
                        iter_id = user_classif_lists['iter_id'][i]

                        both_user_match_dict['src_id'].append(src_id)
                        both_user_match_dict['tutor_class_id'].append(tutor_class_id)
                        both_user_match_dict['iter_id'].append(iter_id)
                        break

        required_user_ids = range(1,50) # skip henrik(0) # [1, 2, 3, 4, 7, 11] # iteration=3
        # First, we will later need a list of all source_ids looked at by users for activelearn iteration:
        all_srcids_for_actlearn_iter = []
        for i, src_id in enumerate(user_classif_lists['src_id']):
            if ((user_classif_lists['user_id'][i] in required_user_ids)):
                if src_id not in all_srcids_for_actlearn_iter:
                    all_srcids_for_actlearn_iter.append(src_id)

        ### Currently we only use the user-consensus sources from the AL*.dat files which were
        #    calculated seperately and once only, and parsed earlier:
        out_dict = {
            'src_id':[],
            'tutor_class_id':[],
            'iter_id':[],
            'all_srcids_for_actlearn_iter':all_srcids_for_actlearn_iter,
            #'both_user_match_dict':both_user_match_dict,
            #'all_srcids_for_actlearn_iter':all_srcids_for_actlearn_iter,
            #'srcid_each_user_makes_some_classification':srcid_each_user_makes_some_classification,
            }


        for session_id in self.user_classifs.keys():
            for iter_id, iter_dict in self.user_classifs[session_id].iteritems():
                for i, src_id in enumerate(iter_dict['src_id']):
                    out_dict['src_id'].append(src_id)
                    out_dict['tutor_class_id'].append(iter_dict['tutor_class_id'][i])
                    out_dict['iter_id'].append(iter_id)
                        
        return out_dict


    def incorporate_user_classified_sources_to_arffstrs(self, datadicts={},
                                                        session_id=0, iteration_id=0,
                                                        n_test_subsample=None):
        """ 1) Given the Active Learning session_id and iteration_id,
               - retrieve all source_ids stored in the DB and related to this session,
               - then get the user classifications for this set of sources.
            2) Combine the initial debosscher training testset with the user-classified sources
               And return and ARFF string with all features for this combined set of training sources

        - In subsequent functions this arff string can then be archived for reference or manual use
        - In subsequent functions this arff string will be used to train a classifier

        OUTPUT: ARFF string of sources with features.
        """
        i_src_from_test_to_skip = []
        srcids_to_addto_trainarffstr = []

        user_classifs = {'user_confident_srcid_list':[],
                         }
        
        if iteration_id > 1:
            ### Query User classifications for all srcids for a session_id, iter_id:
            #   NOTE: ORDER BY user_id, source_id, dtime DESC :
            #      - this is OK since this is the PRIMARY KEY
            #      - The DESC means that the first row occurance for user,srcid is the latest classification
            #   This query retrieves all user-classifications for sources which are given in the previous act_learn iteration_id.
            #      - This means the very first user iteration is iter_id=2, which takes it's subset of actleanr sources from iter_id=1


            ##### This select constrains on act_session & act_iter
            #select_str = """SELECT activelearn_user_class.user_id,
            #                       activelearn_user_class.source_id,
            #                       activelearn_user_class.dtime,
            #                       activelearn_user_class.tutor_class_id,
            #                       activelearn_user_class.other_class_id
            #FROM (SELECT activelearn_actid_srcid.source_id
            #      FROM activelearn_session
            #      JOIN activelearn_actid_srcid ON activelearn_actid_srcid.act_id=activelearn_session.act_id
            #      WHERE activelearn_session.act_session=%d AND
            #            activelearn_session.act_iter=%d) AS T1
            #INNER JOIN activelearn_user_class ON activelearn_user_class.source_id=T1.source_id
            #ORDER BY user_id, source_id, dtime DESC
            #""" % (session_id, iteration_id - 1)

            select_str = """SELECT activelearn_user_class.user_id,
                                   activelearn_user_class.source_id,
                                   activelearn_user_class.dtime,
                                   activelearn_user_class.tutor_class_id,
                                   activelearn_user_class.other_class_id,
                                   T1.act_iter
            FROM (SELECT activelearn_actid_srcid.source_id, activelearn_session.act_iter
                  FROM activelearn_session
                  JOIN activelearn_actid_srcid ON activelearn_actid_srcid.act_id=activelearn_session.act_id
                  WHERE activelearn_session.act_session=%d AND
                        activelearn_session.act_iter > 0) AS T1
            INNER JOIN activelearn_user_class ON activelearn_user_class.source_id=T1.source_id
            ORDER BY user_id, source_id, dtime DESC
            """ % (session_id)

            self.tcp_cursor.execute(select_str)
            results = self.tcp_cursor.fetchall()
            if len(results) == 0:
                raise "ERROR"

            user_classif_lists = {
                'user_id':[],
                'src_id':[],
                'dtime':[],
                'tutor_class_id':[],
                'other_class_id':[],
                'iter_id':[]
                }

            for row in results:
                (user_id, src_id, dtime, tutor_class_id, other_class_id, iter_id) = row
                user_classif_lists['user_id'].append(user_id)
                user_classif_lists['src_id'].append(src_id)
                user_classif_lists['dtime'].append(dtime)
                user_classif_lists['tutor_class_id'].append(tutor_class_id)
                user_classif_lists['other_class_id'].append(other_class_id)
                user_classif_lists['iter_id'].append(iter_id)

            final_user_classifs = self.condense_user_classified_sources(user_classif_lists)

            for i, src_id in enumerate(final_user_classifs['src_id']):
                try:
                    # not sure if this is faster than "if x in y"
                    i_src_test = datadicts['testdata_dict']['srcid_list'].index(str(src_id))
                except:
                    #### These would be sources which users have clasified, but which are not in the ASAS testset
                    print '@@@', src_id
                    raise
                    #continue
                ### So i_src is user classified and in the test-dataset
                
                # This is so that we only get the last classification made by a user for a srcid
                #   - this is more obsolete since final_user_classifs is now an already user-condensed srcid list
                if src_id in srcids_to_addto_trainarffstr:
                    continue # since ORDER(user_id, source_id, dtime), first occurance is most recent user classif
                    # # # # #
                    # # # # # TODO: need to skip other_class!=NULL or ambiguous tutor_class_id classes

                class_id = final_user_classifs['tutor_class_id'][i]
                select_str = "SELECT class_name FROM classes WHERE class_id=%d" % (class_id)
                self.tutor_cursor.execute(select_str)
                results_b = self.tutor_cursor.fetchall()
                if len(results_b) == 0:
                    raise "ERROR"

                i_src_from_test_to_skip.append(i_src_test)
                class_name = results_b[0][0]
                new_arffrow = datadicts['testdata_dict']['arff_rows'][i_src_test] \
                              [:datadicts['testdata_dict']['arff_rows'][i_src_test].rfind(",'")] + \
                              ",'%s'\n" % (class_name)
                srcids_to_addto_trainarffstr.append(src_id)
                ## ## ## datadicts['train_arff_str'] += new_arffrow # this is a MEMORY slow task, could .join() a list
                datadicts['traindata_dict']['arff_rows'].append(new_arffrow)
                datadicts['traindata_dict']['srcid_list'].append(str(src_id))
                datadicts['traindata_dict']['class_list'].append(class_name)
                for feat_name in datadicts['traindata_dict']['featname_longfeatval_dict'].keys():
                    datadicts['traindata_dict']['featname_longfeatval_dict'][feat_name].append(datadicts['testdata_dict']['featname_longfeatval_dict'][feat_name][i_src_test])

            #import pdb; pdb.set_trace()
            #print

        ### Now generate a reduced test-set, minus the user classified sources which were added to the trainset
        new_testset_rows = []
        new_srcid_list = []
        new_class_list = []
        testdata_featname_longfeatval_dict = {}
        for feat_name in datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
            testdata_featname_longfeatval_dict[feat_name] = []
        for i_src, arff_row in enumerate(datadicts['testdata_dict']['arff_rows']):
            if i_src in i_src_from_test_to_skip:
                continue # skip: we dont add to testset these user-consensus sources which we moved to trainset
            new_testset_rows.append(arff_row)
            new_srcid_list.append(datadicts['testdata_dict']['srcid_list'][i_src])
            new_class_list.append(datadicts['testdata_dict']['class_list'][i_src])            
            for feat_name in datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
                testdata_featname_longfeatval_dict[feat_name].append(datadicts['testdata_dict']['featname_longfeatval_dict'][feat_name][i_src])

        datadicts['testdata_dict']['arff_rows'] = new_testset_rows
        datadicts['testdata_dict']['srcid_list'] = new_srcid_list
        datadicts['testdata_dict']['class_list'] = new_class_list
        datadicts['testdata_dict']['featname_longfeatval_dict'] = testdata_featname_longfeatval_dict
        # TODO: (20110407) we no longer want to remove the n_test_subsample number of points here.
        #       will use the full dataset from now on and do partitioning of it within the AL R classification code
        #if n_test_subsample != None:
        #    import random
        #    random.shuffle(new_testset_rows)
        #    new_testset_rows = new_testset_rows[:n_test_subsample] # subselect the 50k down to something smaller for memory issues

        ### Want to plot this later since right now it is all subset sources and not just top active learn chosen sources:
        #self.analysis_plot_period_amp(test_arff_rows=new_testset_rows,
        #                              train_arff_rows=traindata_dict['arff_rows'],
        #                              session_id=session_id,
        #                              iteration_id=iteration_id,
        #                              n_test_subsample=n_test_subsample)

        # KLUDGE: 
        if 0:
            ## ## ##
            if '@data' in datadicts['test_arff_str'][:50000]:
                new_test_arff_str = datadicts['test_arff_str'][:datadicts['test_arff_str'].find('@data')+5]
            else:
                new_test_arff_str = datadicts['test_arff_str'][:datadicts['test_arff_str'].find('@DATA')+5]            
            new_test_arff_str += '\n'.join(new_testset_rows)

        return {#'train_arff_str':'', ## ## ## datadicts['train_arff_str'],
                #'test_arff_str':'', ## ## ## new_test_arff_str,
                #'testdata_featname_longfeatval_dict':testdata_featname_longfeatval_dict,
                #'new_testset_rows':new_testset_rows,
                'all_srcids_for_actlearn_iter':final_user_classifs['all_srcids_for_actlearn_iter'],
                'srcids_to_addto_trainarffstr':srcids_to_addto_trainarffstr,
                'final_user_classifs':final_user_classifs}


    def add_remainder_to_combo(self, combo_result_dict={},
                               remainder_result_dict={},
                               num_srcs_for_users=100):
        """ Add a random selection of "remainder" sources to the dataset

        """

        ###Now randomize the remainder_result_dict index and add to combo_result_dict
        ###   all remaining sources needed to make the expected number of ActLearn user sources (100)

        n_remainders_to_use = num_srcs_for_users - len(self.pars['high_conf_srcids']) - len(combo_result_dict['actlearn_tups'])
        if n_remainders_to_use > len(remainder_result_dict['actlearn_tups']):
            ### NOTE: should only get here during DEBUGGING, when doing only a couple small classifier iterations.
            n_remainders_to_use = len(remainder_result_dict['actlearn_tups'])
        random.shuffle(remainder_result_dict['actlearn_tups'])

        combo_result_dict['actlearn_tups'].extend(remainder_result_dict['actlearn_tups'][:n_remainders_to_use])


    def add_highconfidence_sources_to_combo(self, combo_result_dict={},
                                            remainder_result_dict={},
                                            num_srcs_for_users=100):
        """ Add additional high confidence sources to the list of active learning sources,
              - which will then be added to the database along with the active learned sources

        # TODO: eventually generate this list of srcids from Joey's R algorithm (20110407)
        
        """
        #high_conf_srcids = self.pars['high_conf_srcids'] # This if only for session=0, iter=1
        
        high_conf_srcids_dict = self.rc.get_confident_sources(combo_result_dict=combo_result_dict,
                                                         n_sources_per_class=1)


        # NOTE: he following highconfidence sources should have a couple inserted
        #    at the beginning of the list, then the rest randomly spread throughout.
        # -> todo: maybe do this and the below function in some combining function...


        random.shuffle(high_conf_srcids_dict['source_for_each_class']) # mix them up
        random.shuffle(high_conf_srcids_dict['all_other_sources']) # mix them up
        ### add all but 3 high confidence sources:
        for srcid in high_conf_srcids_dict['source_for_each_class']:
            srcid_str = str(srcid)
            try:
                # NOTE: se should alwyas get here in non DEBUG cases, when all 45k sources are being classified and AL'd.
                ind = combo_result_dict['srcid_list'].index(srcid_str)
                importance = combo_result_dict['err.decr'][ind]
            except:
                importance = -1 # NOTE: eventually all sources should have be active-learn classified and have AL importances, and thus we won't get here.
            combo_result_dict['actlearn_tups'].append((srcid, importance))
        #classifier_dict['actlearn_tups']

        random.shuffle(combo_result_dict['actlearn_tups'])


        ### push the remaining 3 high confidence sources at the beginning:
        for srcid in high_conf_srcids_dict['all_other_sources']:
            srcid_str = str(srcid)
            try:
                # NOTE: se should alwyas get here in non DEBUG cases, when all 45k sources are being classified and AL'd.
                ind = combo_result_dict['srcid_list'].index(srcid_str)
                importance = combo_result_dict['err.decr'][ind]
            except:
                importance = -1 # NOTE: eventually all sources should have be active-learn classified and have AL importances, and thus we won't get here.
            combo_result_dict['actlearn_tups'].insert(0, (srcid, importance))

        #import pdb; pdb.set_trace()
        #print


    def update_training_arffstr_classes(self, arff_datadicts={}):
        """ update classes in the training arff_str
        (specifically debosscher, using TUTOR database query)

        Motivation:
           - arff generated by generate_features_for_several_surveys.py
                either reference xml files or retrieve xmls from http://dotastro.org
                both of which do not give access to the current
                TUTOR database values of table sources.class_id
           - and immediately subsequent functions use the arff_datadicts['train_arff_str']
                rather than the extracted classes.
           - So we need to query the TUTOR database specifically and update these classes in
             the arff string


        """
        if 0:
            ## ## ##
            if '@data' in arff_datadicts['train_arff_str'][:50000]:
                i_header_end = arff_datadicts['train_arff_str'].find('@data')+5
            else:
                i_header_end = arff_datadicts['train_arff_str'].find('@DATA')+5     

            arff_data_str = arff_datadicts['train_arff_str'][i_header_end+1:]
            arff_data_list = arff_data_str.split('\n')

            new_rows = [arff_datadicts['train_arff_str'][:i_header_end]]
            
        new_rows = []
        new_srcid_list = []
        new_class_list = []
        new_featname_longfeatval_dict = {}
        for featname in arff_datadicts['traindata_dict']['featname_longfeatval_dict']:
            new_featname_longfeatval_dict[featname] = []
        for i, arff_row in enumerate(arff_datadicts['traindata_dict']['arff_rows']):
            if len(arff_row) == 0:
                continue
            ## ## ##srcid = int(arff_row[:arff_row.find(',')])
            srcid = int(arff_datadicts['traindata_dict']['srcid_list'][i])
                                
            i_class_begin = arff_row.rfind("'", 0, arff_row.rfind("'")) # this ind of first "'"

            #import pdb; pdb.set_trace()
            #print
            select_str = "select class_name from sources join classes USING (class_id) where source_id=%d" % (srcid)
            self.tutor_cursor.execute(select_str)
            results = self.tutor_cursor.fetchall()
            if len(results) != 1:
                ### This can occur when class_id=0 for a source.
                #   Lets check that this is so and then skip from training on this source.
                select_str = "select class_id from sources where source_id=%d" % (srcid)
                self.tutor_cursor.execute(select_str)
                results = self.tutor_cursor.fetchall()
                if len(results) != 1:
                    raise "ERROR"
                if results[0][0] == 0:
                    continue # this is a training source with class_id=0 in TUTOR, so we skip this source.

            new_class_name = results[0][0]
            ### KLUDGE: for some reason "Beta Persei" is named differntly...
            if new_class_name == "Algol (Beta Persei)":
                new_class_name = "Beta Persei"

            new_arff_row = "%s'%s'" % (arff_row[:i_class_begin], new_class_name)
            arff_datadicts['traindata_dict']['class_list'][i] = new_class_name
            new_rows.append(new_arff_row)
            new_srcid_list.append(str(srcid))
            new_class_list.append(new_class_name)
            # KLUDGY & SLOW:
            for featname in arff_datadicts['traindata_dict']['featname_longfeatval_dict']:
                new_featname_longfeatval_dict[featname].append(arff_datadicts['traindata_dict']['featname_longfeatval_dict'][featname][i])

        arff_datadicts['traindata_dict']['arff_rows'] = new_rows
        arff_datadicts['traindata_dict']['featname_longfeatval_dict'] = new_featname_longfeatval_dict
        arff_datadicts['traindata_dict']['srcid_list'] = new_srcid_list
        arff_datadicts['traindata_dict']['class_list'] = new_class_list
        #info not immediately available# arff_datadicts['traindata_dict']['srcid_list'].append(src_id)
        #info not immediately available# arff_datadicts['traindata_dict']['class_list'].append(class_name)

        if 0:
            ## ## ##
            new_rows.append('\n') # apparently this is needed prior to adding additional ASAS training sources.
            new_train_arff_str = '\n'.join(new_rows)

            arff_datadicts['train_arff_str'] = '' ## ## ## new_train_arff_str




    #############################

    def parse_header_and_attribs_from_arff_str(self, arff_str=''):
        """ parse_header_and_attribs_from_arff_str

        Adapted from rpy2_classifiers.py::insert_missing_value_features()
        """
        feature_list = []
        lines = arff_str.split('\n')
        for i, line in enumerate(lines):
            if line[:10] == '@ATTRIBUTE':
                feat_name = line.split()[1]
                if feat_name not in ['source_id', 'class']:
                    feature_list.append(feat_name)
            elif '@data' in line.lower():
                # we are at the end of the header
                break

        header_str = '\n'.join(lines[:i]) # Does not include @DATA line
        return (header_str, feature_list)

    def combine_train_and_test_datasets(self, orig_arff_datadicts):
        """
        """
        combo_dict = {'srcid_list':[],
                      'class_list':[],
                      'featname_longfeatval_dict':{}}

        combo_dict['srcid_list'].extend(orig_arff_datadicts['testdata_dict']['srcid_list'])
        combo_dict['srcid_list'].extend(orig_arff_datadicts['traindata_dict']['srcid_list'])
        
        combo_dict['class_list'].extend(orig_arff_datadicts['testdata_dict']['class_list'])
        combo_dict['class_list'].extend(orig_arff_datadicts['traindata_dict']['class_list'])

        for featname in orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
            combo_dict['featname_longfeatval_dict'][featname] = []
        
        for featname in orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
            combo_dict['featname_longfeatval_dict'][featname].extend(orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'][featname])
            combo_dict['featname_longfeatval_dict'][featname].extend(orig_arff_datadicts['traindata_dict']['featname_longfeatval_dict'][featname])

        return combo_dict


    def get_crossvalid_errors_for_arffs(self):
        """
        ### Once the embarrassingly_parallelized_imputation() has been run and
        ###    imputed test.arff, train.arff are available, then we generate
        ###    crossvalidated errors for these .arff:

        TODO: want to include arff lines for the AL*dat classifications
               and all ASAS sources contained.
        """
        import glob

        n_iters = 3
        ntree = 1000 # 1000
        mtry = 25 # 25

        self.ipy_tasks = IPython_Task_Administrator()
        self.ipy_tasks.initialize_clients()

        #train_arff_fpaths = ["/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-05-13_04:22:08.073940/source_feats.arff"]
        train_arff_fpaths = ["/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-07-12_23:03:43.667594/source_feats_nocolors.arff"]
        train_arff_glob_str = "/home/pteluser/scratch/active_learn/imputed_arffs/train*_5mtry.arff"
        train_arff_fpaths.extend(glob.glob(train_arff_glob_str))

        for i, train_arff_fpath in enumerate(train_arff_fpaths):
            if i == 0:
                imputed_ntree = 0
                imputed_mtry = 0
            else:
                imputed_ntree = int(train_arff_fpath[train_arff_fpath.rfind("full_")+5:train_arff_fpath.rfind("ntree")])
                imputed_mtry = int(train_arff_fpath[train_arff_fpath.rfind("ntree_")+6:train_arff_fpath.rfind("mtry.")])

            for j_iter in range(n_iters):
                train_arff_str = open(os.path.expandvars(train_arff_fpath)).read()
                traindata_dict = self.rc.parse_full_arff(arff_str=train_arff_str, skip_missingval_lines=False, fill_arff_rows=True)
                if i == 0:
                    #test_arff_fpath = "/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-05-13_07:19:29.802450/source_feats.arff"
                    test_arff_fpath = "/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-07-10_19:38:22.278633/source_feats_nocolors.arff"
                else:
                    test_arff_fpath = train_arff_fpath.replace('/train_full','/test_full')
                test_arff_str = open(os.path.expandvars(test_arff_fpath)).read()
                testdata_dict = self.rc.parse_full_arff(arff_str=test_arff_str, skip_missingval_lines=False, fill_arff_rows=True)

                datadict = {'traindata_dict':traindata_dict,
                            'train_arff_str':train_arff_str,
                            'testdata_dict':testdata_dict,
                            'test_arff_str':test_arff_str}
                incorp_summary = self.incorporate_user_classified_sources_to_arffstrs( \
                                 datadicts=datadict,
                                 session_id=0,
                                 iteration_id=11,
                                 n_test_subsample=0)

                if 0:
                    # for non-ipython testing:
                    import rpy2_classifiers
                    algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
                    rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)
                    error_rate = rc.get_crossvalid_errors(feature_data_dict=datadict['traindata_dict']['featname_longfeatval_dict'], mtry=mtry, ntree=ntree, srcid_list=datadict['traindata_dict']['srcid_list'], class_list=datadict['traindata_dict']['class_list'])
                    import pdb; pdb.set_trace()
                    print
                
                tc_exec_str = """
import rpy2_classifiers
algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)
error_rate = rc.get_crossvalid_errors(feature_data_dict=feature_data_dict, mtry=mtry, ntree=ntree, srcid_list=srcid_list, class_list=class_list)
out_dict = {'error_rate':error_rate,
            'imputed_ntree':imputed_ntree,
            'imputed_mtry':imputed_mtry,
            'iter':iter}
                """
                taskid = self.ipy_tasks.tc.run(self.ipy_tasks.kernel_client.StringTask(tc_exec_str,
                                           push={'feature_data_dict':datadict['traindata_dict']['featname_longfeatval_dict'],
                                                 'ntree':ntree,
                                                 'mtry':mtry,
                                                 'iter':j_iter,
                                                 'imputed_ntree':imputed_ntree,
                                                 'imputed_mtry':imputed_mtry,
                                                 'srcid_list':datadict['traindata_dict']['srcid_list'],
                                                 'class_list':datadict['traindata_dict']['class_list'],
                                                 },
                                  pull='out_dict', 
                                  retries=3))
                self.ipy_tasks.task_id_list.append(taskid)

        import pdb; pdb.set_trace()
        print
        import numpy
        out_tups = []
        tup_dict = {}
        for taskid in self.ipy_tasks.task_id_list:
            temp = self.ipy_tasks.tc.get_task_result(taskid, block=False)
            results = temp['out_dict']
            tup = (temp['out_dict']['imputed_mtry'], temp['out_dict']['imputed_ntree'])
            if not tup_dict.has_key(tup):
                tup_dict[tup] = []
            tup_dict[tup].append(temp['out_dict']['error_rate'])

        tup_list_sorted = tup_dict.keys()
        tup_list_sorted.sort()
        final_tups = []
        for a_tup in tup_list_sorted:
            print a_tup[0], a_tup[1], numpy.mean(tup_dict[a_tup]), numpy.std(tup_dict[a_tup])
            final_tups.append((a_tup[0], a_tup[1], numpy.mean(tup_dict[a_tup]), numpy.std(tup_dict[a_tup])))
        import matplotlib.pyplot as pyplot
        y = numpy.array(final_tups)
        pyplot.plot(y[:,1], [y[:,2][0]]*len(y[:,1]))
        pyplot.errorbar(y[:,1], y[:,2], yerr=y[:,3], fmt='ro')
        pyplot.savefig("/home/pteluser/scratch/color_imputed_errorrate_vs_ntree_mtry5.png")

        import pdb; pdb.set_trace()
        print




    def embarrassingly_parallelized_imputation(self, feature_data_dict={},
                                               srcid_list=[],
                                               class_list=[],
                                               arff_str='',
                                               train_srcids=[]):
        """ Do imputation of missing-value feature values in dataset

          - See arxiv 1105.0828v1 for more information on
                 the R:MissForest imputation code for R:randomForest()

        Adapted from rpy2_classifiers.py:imputation_using_missForest()

        KLUDGE: Since this function accesses self.rc.* ideally it will be
                 contained in rpy2_classifiers.py
           - although, it will contain Ipython parallelization code...

        NOTE: the input sources are now expected to be a combination of test and training
              since the color imputation works just as well combining these together.
          - this means that the rc.generate_imputed_arff_for_ntree() needs to be able
            to split and write seperate arff for test and training data.
           
        """
        # TODO: parse header and ordered_attribs from the arff_str
        (header_str, feature_list) = self.parse_header_and_attribs_from_arff_str(arff_str=arff_str)


        self.ipy_tasks = IPython_Task_Administrator()
        self.ipy_tasks.initialize_clients()

        # TODO: eventually the following will iterate over a bunch of ntree values

        mtry = 5

        if 0:
            # for non-ipython testing:
            ntree = 2
            import rpy2_classifiers
            algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
            rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)
            arff_fpath_dict = rc.generate_imputed_arff_for_ntree(feature_data_dict, mtry=mtry, ntree=ntree, header_str=header_str, feature_list=feature_list, srcid_list=srcid_list, class_list=class_list, train_srcids=train_srcids)
            import pdb; pdb.set_trace()
            print

        #ntree_list = range(10,200,5) # just get some fast arff files created
        ntree_list = range(10,250,10)
        #ntree_list = range(2,20,1)
        #ntree_list.extend(range(20,410,10))
        
        for ntree in ntree_list:
            tc_exec_str = """
algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)
out_dict = rc.generate_imputed_arff_for_ntree(feature_data_dict, mtry=mtry, ntree=ntree, header_str=header_str, feature_list=feature_list, srcid_list=srcid_list, class_list=class_list, train_srcids=train_srcids)
                """
            taskid = self.ipy_tasks.tc.run(self.ipy_tasks.kernel_client.StringTask(tc_exec_str,
                                           push={'feature_data_dict':feature_data_dict,
                                                 'ntree':ntree,
                                                 'mtry':mtry,
                                                 'header_str':header_str,
                                                 'feature_list':feature_list,
                                                 'srcid_list':srcid_list,
                                                 'class_list':class_list,
                                                 'train_srcids':train_srcids},
                                  pull='out_dict', 
                                  retries=3))
            self.ipy_tasks.task_id_list.append(taskid)

        # debug:
        import pdb; pdb.set_trace()
        print
        temp = self.ipy_tasks.tc.get_task_result(taskid, block=False)
        results = temp['out_dict']
        ### TODO: just see if these tasks run without fail so far
        #     - then look at arff_generateion_master.py L2186 for task pulling
        import pdb; pdb.set_trace()
        print

    #############################



    # This doesn't work when accessing robject.global or robject.r
    #  - seems there are inconsistancies with retrieving rpy2 objects witin ipython tasks
    #  - it is apparently on rpy2's side where the segfault is triggered, not ipython
    #https://github.com/ipython/ipython/issues/434
    #http://www.google.com/search?sourceid=chrome&ie=UTF-8&q=rpy2+robjects+ipython+parallel
    #  - I encountered this problem on betsy R v2.19
    #     - could not retrieve R computed values in activelearn_utils.py:parallelized_imputation()
    #     - although, I think this was partially due to trying to retrieve deeply nested variables reather than just the final R-computed classifications
    def parallelized_imputation(self, feature_data_dict, mtry=None, ntree=None):
        """ Do imputation of missing-value feature values in dataset

          - See arxiv 1105.0828v1 for more information on
                 the R:MissForest imputation code for R:randomForest()

        Adapted from rpy2_classifiers.py:imputation_using_missForest()

        KLUDGE: Since this function accesses self.rc.* ideally it will be
                 contained in rpy2_classifiers.py
           - although, it will contain Ipython parallelization code...
           
        """
        self.ipy_tasks = IPython_Task_Administrator()
        self.ipy_tasks.initialize_clients()

        from rpy2.robjects.packages import importr
        from rpy2 import robjects

        import rpy2_classifiers
        
        r_data_dict = {}
        for feat_name, feat_longlist in feature_data_dict.iteritems():
            r_data_dict[feat_name] = robjects.FloatVector(feat_longlist)
        features_r_data = robjects.r['data.frame'](**r_data_dict)

        robjects.globalenv['xmis'] = features_r_data
        robjects.globalenv['maxiter'] = 10
        robjects.globalenv['decreasing'] = 'FALSE'
        robjects.globalenv['verbose'] = 'TRUE'
        robjects.globalenv['mtry'] = mtry
        ntree = 5
        robjects.globalenv['ntree'] = ntree

        #n = len(feature_data_dict[feature_data_dict.keys()[0]]) #robjects.r("nrow(xmis)")
        p = len(feature_data_dict) # robjects.r("ncol(xmis)")

        ### NOTE: xtrue could be a complete data matrix, without missing values.
        #          - if defined, the missforest code will generate errors in the imputation
        robjects.r("""
            n <- nrow(xmis)
            p <- ncol(xmis)
            xtrue <- NA
            ## perform initial guess on xmis
            ximp <- xmis
            xAttrib <- lapply(xmis, attributes)
            varType <- character(p)
            for (t.co in 1:p){
              if (is.null(xAttrib[[t.co]])){
                varType[t.co] <- 'numeric'
                ximp[is.na(xmis[,t.co]),t.co] <- mean(xmis[,t.co], na.rm = TRUE)
              } else {
                varType[t.co] <- xAttrib[[t.co]]$class
                ## take the level which is more 'likely' (majority vote)
                max.level <- max(table(ximp[,t.co]))
                ## if there are several classes which are major, sample one at random
                class.assign <- sample(names(which(max.level == summary(ximp[,t.co]))), 1)
                ## it shouldn't be the NA class
                if (class.assign != "NA's"){
                  ximp[is.na(xmis[,t.co]),t.co] <- class.assign
                } else {
                  while (class.assign == "NA's"){
                    class.assign <- sample(names(which(max.level ==
                                                       summary(ximp[,t.co]))), 1)
                  }
                  ximp[is.na(xmis[,t.co]),t.co] <- class.assign
                }
              }
            }

            ## extract missingness pattern
            NAloc <- is.na(xmis)            # where are missings
            noNAvar <- apply(NAloc, 2, sum) # how many are missing in the vars
            sort.j <- order(noNAvar) # indices of increasing amount of NA in vars
            if (decreasing)
              sort.j <- rev(sort.j)
            sort.noNAvar <- noNAvar[sort.j]
             
            ## output
            Ximp <- vector('list', maxiter)
  
            ## initialize parameters of interest
            iter <- 0
            k <- length(unique(varType))
            convNew <- rep(0, k)
            convOld <- rep(Inf, k)
            OOBerror <- numeric(p)
            names(OOBerror) <- varType

            ## setup convergence variables w.r.t. variable types
            if (k == 1){
              if (unique(varType) == 'numeric'){
                names(convNew) <- c('numeric')
              } else {
                names(convNew) <- c('factor')
              }
              convergence <- c()
              OOBerr <- numeric(1)
            } else {
              names(convNew) <- c('numeric', 'factor')
              convergence <- matrix(NA, ncol = 2)
              OOBerr <- numeric(2)
            }

            ## function to yield the stopping criterion in the following 'while' loop
            stopCriterion <- function(varType, convNew, convOld, iter, maxiter){
              k <- length(unique(varType))
              if (k == 1){
                  (convNew < convOld) & (iter < maxiter)
              } else {
                ((convNew[1] < convOld[1]) | (convNew[2] < convOld[2])) & (iter < maxiter)
              }
            }
        """)

        if 0:
            ### Do xtrue imputation error calculation later, so load the following functions:
            robjects.r("""
                varClass <- function(x){
                  xAttrib <- lapply(x, attributes)
                  p <- ncol(x)
                  x.types <- character(p)
                  for (t.co in 1:p){
                    if (is.null(xAttrib[[t.co]])){
                      x.types[t.co] <- 'numeric'
                    } else {
                      x.types[t.co] <- xAttrib[[t.co]]$class
                    }
                  }
                  return(x.types)
                }


                mixError <- function(ximp, xmis, xtrue)
                {
                  ## Purpose:
                  ## Calculates the difference between to matrices. For all numeric
                  ## variables the NRMSE is used and for all categorical variables
                  ## the relative number of false entries is returned.
                  ## ----------------------------------------------------------------------
                  ## Arguments:
                  ## ximp      = (imputed) matrix
                  ## xmis      = matrix with missing values
                  ## xtrue     = true matrix (or any matrix to be compared with ximp)
                  ## ----------------------------------------------------------------------
                  ## Author: Daniel Stekhoven, Date: 26 Jul 2010, 10:10

                  x.types <- varClass(ximp)
                  n <- nrow(ximp)
                  k <- length(unique(x.types))
                  err <- rep(Inf, k)
                  t.co <- 1
                  if (k == 1){
                    if (unique(x.types) == 'numeric'){
                      names(err) <- c('numeric')
                    } else {
                      names(err) <- c('factor')
                      t.co <- 1
                    }
                  } else {
                    names(err) <- c('numeric', 'factor')
                    t.co <- 2
                  }
                  for (t.type in names(err)){
                    t.ind <- which(x.types == t.type)
                    if (t.type == "numeric"){
                      err[1] <- nrmse(ximp[,t.ind], xmis[,t.ind], xtrue[,t.ind])
                    } else {
                      dist <- sum(as.character(as.matrix(ximp[,t.ind])) != as.character(as.matrix(xtrue[,t.ind])))
                      no.na <- sum(is.na(xmis[,x.types == 'factor']))
                      if (no.na == 0){
                        err[t.co] <- 0
                      } else {
                        err[t.co] <- dist / no.na
                      }
                    }
                  }
                  return(err)
                }


                nrmse <- function(ximp, xmis, xtrue){
                  mis <- is.na(xmis)
                  sqrt(mean((ximp[mis]-xtrue[mis])^{2})/var(xtrue[mis]))
                }


                prodNA <- function(x, noNA = 0.1){
                  n <- nrow(x)
                  p <- ncol(x)
                  NAloc <- rep(FALSE, n*p)
                  NAloc[sample(n*p, floor(n*p*noNA))] <- TRUE
                  x[matrix(NAloc, nrow = n, ncol = p)] <- NA
                  return(x)
                }
            """)

        #import pdb; pdb.set_trace()
        #print
        continue_loop = True 
        i_loop = 0
        while (continue_loop):
            print "i_loop:", i_loop
            i_loop += 1
            robjects.r("""
                if (iter != 0){
                  convOld <- convNew
                  OOBerrOld <- OOBerr
                }
                cat("  missForest iteration", iter+1, "in progress...")
                t.start <- proc.time()
                ximp.old <- ximp
                """)
            for s in range(1,p):
                ### TODO: parallelize the following:
                robjects.r("""
                    varInd <- sort.j[%d]
                    has_na <- noNAvar[[varInd]]
                    obsi <- !NAloc[,varInd] # which i's are observed
                    misi <- NAloc[,varInd] # which i's are missing
                    """ % (s))
                has_na = robjects.r("has_na")
                print s, list(has_na)[0]
                if list(has_na)[0] == 0:
                    continue # since this is a feature that has no NA values, we do not impute.
                obsi = list(numpy.array(robjects.r("obsi"), dtype=numpy.bool)) #numpy.array(robjects.r("obsi"), dtype=numpy.int)
                misi = list(numpy.array(robjects.r("misi"), dtype=numpy.bool)) #numpy.array(robjects.r("misi"), dtype=numpy.int)
                varInd = list(numpy.array(robjects.r("sort.j[%d]" % (s)), dtype=numpy.int)) #numpy.array(robjects.r("sort.j[%d]" % (s))) # 2
                varType = list(numpy.array(robjects.r("varType"), dtype=numpy.str)) # numpy.array(robjects.r("varType"))

                #import pdb; pdb.set_trace()
                #print
                # TODO: convert the following into ndarray?
                ximp = numpy.array(robjects.r("ximp")).T # NOTE:rpy2 expects the 2D array to hhave col/rows reversed.   ximp[p2p_scatter_over_mad]=[sources]

                # TODO: look at generate_weka_classifiers.py:L1376 to do IPython task()
                #(misY, OOBerror_val) = rpy2_classifiers.missforest_parallel_task(varInd, ximp, obsi, misi, varType, ntree, p)
                tc_exec_str = """
out_dict = rpy2_classifiers.missforest_parallel_task(varInd, ximp, obsi, misi, varType, ntree, p)
                """
                """
(misY, OOBerror_val) = rpy2_classifiers.missforest_parallel_task(varInd, ximp, obsi, misi, varType, ntree)
out_dict = {'misY':misY,
            'OOBerror_val':OOBerror_val}
                """
                taskid = self.ipy_tasks.tc.run(self.ipy_tasks.kernel_client.StringTask(tc_exec_str,
                                                   push={'varInd':varInd,
                                                         'ximp':ximp,
                                                         'obsi':obsi,
                                                         'misi':misi,
                                                         'varType':varType,
                                                         'ntree':ntree,
                                                         'p':p},
                                          pull='out_dict', 
                                          retries=3))
                self.ipy_tasks.task_id_list.append(taskid)

                # debug:
                import pdb; pdb.set_trace()
                print
                temp = self.ipy_tasks.tc.get_task_result(taskid, block=False)
                results = temp.get('out_dict',None)
                ### TODO: just see if these tasks run without fail so far
                #     - then look at arff_generateion_master.py L2186 for task pulling
                import pdb; pdb.set_trace()
                print
                
                #import pdb; pdb.set_trace()
                #print
                robjects.globalenv['misY'] = misY
                robjects.globalenv['OOBerror_val'] = OOBerror_val
                robjects.r("""
                    varInd <- sort.j[%d]
                    misi <- NAloc[,varInd] # which i's are missing
                    ## replace old imputed value with prediction
                    ximp[misi, varInd] <- misY
                    OOBerror[varInd] <- OOBerror_val
                """ % (s))
            import pdb; pdb.set_trace()
            print
            print "Done with s-loop, Now we determine the convergence..."
            ### Now we determine the convergence
            #import pdb; pdb.set_trace()
            #print
            robjects.r("""
                iter <- iter+1
                Ximp[[iter]] <- ximp
    
                t.co2 <- 1
                ## check the difference between iteration steps
                for (t.type in names(convNew)){
                  t.ind <- which(varType == t.type)
                  if (t.type == "numeric"){
                    convNew[t.co2] <- sum((ximp[,t.ind]-ximp.old[,t.ind])^2)/sum(ximp[,t.ind]^2)
                  } else {
                    dist <- sum(as.character(as.matrix(ximp[,t.ind])) != as.character(as.matrix(ximp.old[,t.ind])))
                    convNew[t.co2] <- dist / (n * sum(varType == 'factor'))
                  }
                  t.co2 <- t.co2 + 1
                }

                ## compute estimated imputation error
                NRMSE <- sqrt(mean(OOBerror[varType=='numeric'])/var(as.vector(as.matrix(xmis[,varType=='numeric'])), na.rm = TRUE))
                PFC <- mean(OOBerror[varType=='factor'])
                if (k==1){
                  if (unique(varType)=='numeric'){
                    OOBerr <- NRMSE
                    names(OOBerr) <- 'NRMSE'
                  } else {
                    OOBerr <- PFC
                    names(OOBerr) <- 'PFC'
                  }
                } else {
                  OOBerr <- c(NRMSE, PFC)
                  names(OOBerr) <- c('NRMSE', 'PFC')
                }

                ## return status output, if desired
                if (verbose){
                  delta.start <- proc.time() - t.start
                  if (any(!is.na(xtrue))){
                    err <- mixError(ximp, xmis, xtrue)
                    cat("    error(s):", err, "\n")
                  }
                  cat("    estimated error(s):", OOBerr, "\n")
                  cat("    convergence:", convNew, "\n")
                  cat("    time:", delta.start[3], "seconds\n\n")
                }
                continue_loop <- stopCriterion(varType, convNew, convOld, iter, maxiter)
            """)
            continue_loop = list(robjects.r("continue_loop"))[0]
            import pdb; pdb.set_trace()
            print
        robjects.r("""
            if (iter == maxiter){
              miss_data.imp <- list(Ximp = Ximp[[iter]], error = OOBerr)
            } else {
              miss_data.imp <- list(Ximp = Ximp[[iter-1]], error = OOBerrOld)
            }
        """)

        feature_names = list(robjects.r("names(miss_data.imp$Ximp)"))

        out_feat_dict = {}
        for i, feat_name in enumerate(feature_names):
            out_feat_dict[feat_name] = list(robjects.r("miss_data.imp$Ximp$%s" % (feat_name)))
        import pdb; pdb.set_trace()
        print
        return out_feat_dict


    def parallelized_imputation__works_single(self, feature_data_dict, mtry=None, ntree=None):
        """ Do imputation of missing-value feature values in dataset

          - See arxiv 1105.0828v1 for more information on
                 the R:MissForest imputation code for R:randomForest()

        Adapted from rpy2_classifiers.py:imputation_using_missForest()

        KLUDGE: Since this function accesses self.rc.* ideally it will be
                 contained in rpy2_classifiers.py
           - although, it will contain Ipython parallelization code...
           
        """
        from rpy2.robjects.packages import importr
        from rpy2 import robjects

        r_data_dict = {}
        for feat_name, feat_longlist in feature_data_dict.iteritems():
            r_data_dict[feat_name] = robjects.FloatVector(feat_longlist)
        features_r_data = robjects.r['data.frame'](**r_data_dict)

        robjects.globalenv['xmis'] = features_r_data
        robjects.globalenv['maxiter'] = 10
        robjects.globalenv['decreasing'] = 'FALSE'
        robjects.globalenv['verbose'] = 'TRUE'
        robjects.globalenv['mtry'] = mtry
        robjects.globalenv['ntree'] = 5 #ntree

        #n = len(feature_data_dict[feature_data_dict.keys()[0]]) #robjects.r("nrow(xmis)")
        p = len(feature_data_dict) # robjects.r("ncol(xmis)")

        ### NOTE: xtrue could be a complete data matrix, without missing values.
        #          - if defined, the missforest code will generate errors in the imputation
        robjects.r("""
            n <- nrow(xmis)
            p <- ncol(xmis)
            xtrue <- NA
            ## perform initial guess on xmis
            ximp <- xmis
            xAttrib <- lapply(xmis, attributes)
            varType <- character(p)
            for (t.co in 1:p){
              if (is.null(xAttrib[[t.co]])){
                varType[t.co] <- 'numeric'
                ximp[is.na(xmis[,t.co]),t.co] <- mean(xmis[,t.co], na.rm = TRUE)
              } else {
                varType[t.co] <- xAttrib[[t.co]]$class
                ## take the level which is more 'likely' (majority vote)
                max.level <- max(table(ximp[,t.co]))
                ## if there are several classes which are major, sample one at random
                class.assign <- sample(names(which(max.level == summary(ximp[,t.co]))), 1)
                ## it shouldn't be the NA class
                if (class.assign != "NA's"){
                  ximp[is.na(xmis[,t.co]),t.co] <- class.assign
                } else {
                  while (class.assign == "NA's"){
                    class.assign <- sample(names(which(max.level ==
                                                       summary(ximp[,t.co]))), 1)
                  }
                  ximp[is.na(xmis[,t.co]),t.co] <- class.assign
                }
              }
            }

            ## extract missingness pattern
            NAloc <- is.na(xmis)            # where are missings
            noNAvar <- apply(NAloc, 2, sum) # how many are missing in the vars
            sort.j <- order(noNAvar) # indices of increasing amount of NA in vars
            if (decreasing)
              sort.j <- rev(sort.j)
            sort.noNAvar <- noNAvar[sort.j]
             
            ## output
            Ximp <- vector('list', maxiter)
  
            ## initialize parameters of interest
            iter <- 0
            k <- length(unique(varType))
            convNew <- rep(0, k)
            convOld <- rep(Inf, k)
            OOBerror <- numeric(p)
            names(OOBerror) <- varType

            ## setup convergence variables w.r.t. variable types
            if (k == 1){
              if (unique(varType) == 'numeric'){
                names(convNew) <- c('numeric')
              } else {
                names(convNew) <- c('factor')
              }
              convergence <- c()
              OOBerr <- numeric(1)
            } else {
              names(convNew) <- c('numeric', 'factor')
              convergence <- matrix(NA, ncol = 2)
              OOBerr <- numeric(2)
            }

            ## function to yield the stopping criterion in the following 'while' loop
            stopCriterion <- function(varType, convNew, convOld, iter, maxiter){
              k <- length(unique(varType))
              if (k == 1){
                  (convNew < convOld) & (iter < maxiter)
              } else {
                ((convNew[1] < convOld[1]) | (convNew[2] < convOld[2])) & (iter < maxiter)
              }
            }
        """)

        if 0:
            ### Do xtrue imputation error calculation later, so load the following functions:
            robjects.r("""
                varClass <- function(x){
                  xAttrib <- lapply(x, attributes)
                  p <- ncol(x)
                  x.types <- character(p)
                  for (t.co in 1:p){
                    if (is.null(xAttrib[[t.co]])){
                      x.types[t.co] <- 'numeric'
                    } else {
                      x.types[t.co] <- xAttrib[[t.co]]$class
                    }
                  }
                  return(x.types)
                }


                mixError <- function(ximp, xmis, xtrue)
                {
                  ## Purpose:
                  ## Calculates the difference between to matrices. For all numeric
                  ## variables the NRMSE is used and for all categorical variables
                  ## the relative number of false entries is returned.
                  ## ----------------------------------------------------------------------
                  ## Arguments:
                  ## ximp      = (imputed) matrix
                  ## xmis      = matrix with missing values
                  ## xtrue     = true matrix (or any matrix to be compared with ximp)
                  ## ----------------------------------------------------------------------
                  ## Author: Daniel Stekhoven, Date: 26 Jul 2010, 10:10

                  x.types <- varClass(ximp)
                  n <- nrow(ximp)
                  k <- length(unique(x.types))
                  err <- rep(Inf, k)
                  t.co <- 1
                  if (k == 1){
                    if (unique(x.types) == 'numeric'){
                      names(err) <- c('numeric')
                    } else {
                      names(err) <- c('factor')
                      t.co <- 1
                    }
                  } else {
                    names(err) <- c('numeric', 'factor')
                    t.co <- 2
                  }
                  for (t.type in names(err)){
                    t.ind <- which(x.types == t.type)
                    if (t.type == "numeric"){
                      err[1] <- nrmse(ximp[,t.ind], xmis[,t.ind], xtrue[,t.ind])
                    } else {
                      dist <- sum(as.character(as.matrix(ximp[,t.ind])) != as.character(as.matrix(xtrue[,t.ind])))
                      no.na <- sum(is.na(xmis[,x.types == 'factor']))
                      if (no.na == 0){
                        err[t.co] <- 0
                      } else {
                        err[t.co] <- dist / no.na
                      }
                    }
                  }
                  return(err)
                }


                nrmse <- function(ximp, xmis, xtrue){
                  mis <- is.na(xmis)
                  sqrt(mean((ximp[mis]-xtrue[mis])^{2})/var(xtrue[mis]))
                }


                prodNA <- function(x, noNA = 0.1){
                  n <- nrow(x)
                  p <- ncol(x)
                  NAloc <- rep(FALSE, n*p)
                  NAloc[sample(n*p, floor(n*p*noNA))] <- TRUE
                  x[matrix(NAloc, nrow = n, ncol = p)] <- NA
                  return(x)
                }
            """)

        #import pdb; pdb.set_trace()
        #print
        continue_loop = True 
        i_loop = 0
        while (continue_loop):
            print "i_loop:", i_loop
            i_loop += 1
            robjects.r("""
                if (iter != 0){
                  convOld <- convNew
                  OOBerrOld <- OOBerr
                }
                cat("  missForest iteration", iter+1, "in progress...")
                t.start <- proc.time()
                ximp.old <- ximp
                """)

            for s in range(1,p):
                #print "(Py: s:", s
                ### TODO: parallelize the following:

                # NODE INPUT: ximp, sort.j[s], noNAvar, ntree, OOBerror

                robjects.globalenv['s'] = s
                robjects.r("""
                    varInd <- sort.j[s]
                    if (noNAvar[[varInd]] != 0){
                      cat("s:", s, "\n")  
                      obsi <- !NAloc[,varInd] # which i's are observed
                      misi <- NAloc[,varInd] # which i's are missing
                      obsY <- ximp[obsi, varInd] # training response
                      obsX <- ximp[obsi, seq(1, p)[-varInd]] # training variables
                      misX <- ximp[misi, seq(1, p)[-varInd]] # prediction variables
                      typeY <- varType[varInd]
                      if (typeY == 'numeric'){
                        ## train random forest on observed data
                        cat("    B4 randomForest()", "\n")
                        RF <- randomForest(x = obsX, y = obsY, ntree = ntree)
                        ## record out-of-bag error
                        OOBerror[varInd] <- RF$mse[ntree]
                        ## predict missing values in column varInd
                        cat("    B4 predict()", "\n")
                        misY <- predict(RF, misX)
                      } else { # if Y is categorical          
                        obsY <- factor(obsY) ## remove empty classes
                        summarY <- summary(obsY)
                        if (length(summarY) == 1){ ## if there is only one level left
                          cat("    B4 factor(rep(names(summarY)", "\n")
                          misY <- factor(rep(names(summarY), sum(misi)))
                        } else {
                          ## train random forest on observed data
                          cat("    B4 randomForest() !numeric", "\n")
                          RF <- randomForest(x = obsX, y = obsY, ntree = ntree)
                          ## record out-of-bag error
                          OOBerror[varInd] <- RF$err.rate[[ntree,1]]
                          ## predict missing values in column varInd
                          cat("    B4 randomForest() !predict", "\n")
                          misY <- predict(RF, misX)
                        }
                      }
                      ## replace old imputed value with prediction
                      ximp[misi, varInd] <- misY
                    }
                """)
                if s == 64:
                    import pdb; pdb.set_trace()
                    print
            print "Done with s-loop, Now we determine the convergence..."
            ### Now we determine the convergence
            #import pdb; pdb.set_trace()
            #print
            robjects.r("""
                iter <- iter+1
                Ximp[[iter]] <- ximp
    
                t.co2 <- 1
                ## check the difference between iteration steps
                for (t.type in names(convNew)){
                  t.ind <- which(varType == t.type)
                  if (t.type == "numeric"){
                    convNew[t.co2] <- sum((ximp[,t.ind]-ximp.old[,t.ind])^2)/sum(ximp[,t.ind]^2)
                  } else {
                    dist <- sum(as.character(as.matrix(ximp[,t.ind])) != as.character(as.matrix(ximp.old[,t.ind])))
                    convNew[t.co2] <- dist / (n * sum(varType == 'factor'))
                  }
                  t.co2 <- t.co2 + 1
                }

                ## compute estimated imputation error
                NRMSE <- sqrt(mean(OOBerror[varType=='numeric'])/var(as.vector(as.matrix(xmis[,varType=='numeric'])), na.rm = TRUE))
                PFC <- mean(OOBerror[varType=='factor'])
                if (k==1){
                  if (unique(varType)=='numeric'){
                    OOBerr <- NRMSE
                    names(OOBerr) <- 'NRMSE'
                  } else {
                    OOBerr <- PFC
                    names(OOBerr) <- 'PFC'
                  }
                } else {
                  OOBerr <- c(NRMSE, PFC)
                  names(OOBerr) <- c('NRMSE', 'PFC')
                }

                ## return status output, if desired
                if (verbose){
                  delta.start <- proc.time() - t.start
                  if (any(!is.na(xtrue))){
                    err <- mixError(ximp, xmis, xtrue)
                    cat("    error(s):", err, "\n")
                  }
                  cat("    estimated error(s):", OOBerr, "\n")
                  cat("    convergence:", convNew, "\n")
                  cat("    time:", delta.start[3], "seconds\n\n")
                }
                continue_loop <- stopCriterion(varType, convNew, convOld, iter, maxiter)
            """)
            continue_loop = list(robjects.r("continue_loop"))[0]
            #import pdb; pdb.set_trace()
            #print
        robjects.r("""
            if (iter == maxiter){
              miss_data.imp <- list(Ximp = Ximp[[iter]], error = OOBerr)
            } else {
              miss_data.imp <- list(Ximp = Ximp[[iter-1]], error = OOBerrOld)
            }
        """)

        feature_names = list(robjects.r("names(miss_data.imp$Ximp)"))

        out_feat_dict = {}
        for i, feat_name in enumerate(feature_names):
            out_feat_dict[feat_name] = list(robjects.r("miss_data.imp$Ximp$%s" % (feat_name)))
        import pdb; pdb.set_trace()
        print
        return out_feat_dict


    def apply_imputation_to_train_and_test_data(self, traindata_dict={}, testdata_dict={},
                                                mtry=5, ntrees=500, nodesize=5):
        """ We need to apply imputation algorithms when working with features which
        may have missing values for some sources.  For example, NOMAD-color difference features.
        
        NOTE: we do these imputations to both training and testing data since sources
              with missing value features are found in both data sets.

          - See arxiv 1105.0828v1 for more information on
                 the R:MissForest imputation code for R:randomForest()
        """
        if 1:
            #### R / Python parallelized, hybrid imputation code:
            #  - seems there are inconsistancies with retrieving rpy2 objects witin ipython tasks
            #  - it is apparently on rpy2's side where the segfault is triggered, not ipython
            #https://github.com/ipython/ipython/issues/434
            #http://www.google.com/search?sourceid=chrome&ie=UTF-8&q=rpy2+robjects+ipython+parallel
            #  - I encountered this problem on betsy R v2.19
            #     - could not retrieve R computed values in activelearn_utils.py:parallelized_imputation()
            #     - although, I think this was partially due to trying to retrieve deeply nested variables reather than just the final R-computed classifications
            #### (Testing-dataset):
            if 0:
                new_features_r_data = self.parallelized_imputation( \
                                                         testdata_dict['featname_longfeatval_dict'],
                                                         mtry=mtry, ntree=ntrees)
                testdata_dict['featname_longfeatval_dict'] = new_features_r_data
        else:
            #### Non-parallel Testing data:
            new_features_r_data = self.rc.imputation_using_missForest(testdata_dict['featname_longfeatval_dict'],
                                                                      mtry=mtry, ntree=ntrees)
            testdata_dict['featname_longfeatval_dict'] = new_features_r_data
        
        ### Training data:
        new_features_r_data = self.rc.imputation_using_missForest(traindata_dict['featname_longfeatval_dict'],
                                                                  mtry=mtry, ntree=ntrees)
        traindata_dict['featname_longfeatval_dict'] = new_features_r_data


    def get_cost_freqsignif_info(self, srcid_list=[], traindata_dict={}, testdata_dict={}, all_srcids_for_actlearn_iter=[]):
        """ Get freq_signif feature information of user-classified sources,
        for use in generating costs.
        """
        both_user_match_srcid_bool = []
        actlearn_sources_freqsignifs = []
        #for srcid in new_arffstrs_dict['final_user_classifs']['src_id']:
        temp_srcid_list = []

        # I want a set of all sources in either list:
        #  - srcid_list : 533)iter=11) : this is all sources found in AL_*.dat)
        #  - all_srcids_for_actlearn_iter : these are all sources stored in database to be shown to ALLStars users
        srcid_union_list = list(set(srcid_list) | set(all_srcids_for_actlearn_iter))
        ## ## ##for srcid in new_arffstrs_dict['final_user_classifs']['all_srcids_for_actlearn_iter']:
        #for srcid in srcid_list:
        for srcid in srcid_union_list:
            srcid_str = str(srcid)
            if srcid_str in traindata_dict['srcid_list']:
                i_src = traindata_dict['srcid_list'].index(srcid_str)
                freqsignif = traindata_dict['featname_longfeatval_dict']['freq_signif'][i_src]
            else:
                try:
                    i_src = testdata_dict['srcid_list'].index(srcid_str)
                    freqsignif = testdata_dict['featname_longfeatval_dict']['freq_signif'][i_src]
                except:
                    print '! NEED to ADD to reduce_testset_using_costs() OK list', srcid_str
            actlearn_sources_freqsignifs.append(freqsignif)
            #if srcid in new_arffstrs_dict['final_user_classifs']['both_user_match_dict']['src_id']:
            #STILLCRAP#if srcid in new_arffstrs_dict['final_user_classifs']['srcid_each_user_makes_some_classification']:
            ### It looks like this was also done of iteration_id=2:
            if srcid in srcid_list:
                both_user_match_srcid_bool.append(True)
                temp_srcid_list.append(srcid)
            else:
                both_user_match_srcid_bool.append(False)
                temp_srcid_list.append(srcid)

        for k in range(len(temp_srcid_list)):
            #if temp_srcid_list[i] in new_arffstrs_dict['final_user_classifs']['src_id']:
            print k, temp_srcid_list[k], both_user_match_srcid_bool[k], actlearn_sources_freqsignifs[k]

        return (both_user_match_srcid_bool, actlearn_sources_freqsignifs)
    

    def shuffle_srcid_rows(self, datadict={}):
        """ Shuffle the srcids / rows
        """
        inds = range(len(datadict['srcid_list']))
        random.shuffle(inds)

        new_arff_rows = []
        new_srcid_list = []
        new_class_list = []
        new_featname_longfeatval_dict = {}
        for feat_name in datadict['featname_longfeatval_dict'].keys():
            new_featname_longfeatval_dict[feat_name] = []
            
        for i in inds:
            new_arff_rows.append(datadict['arff_rows'][i])
            new_srcid_list.append(datadict['srcid_list'][i])
            new_class_list.append(datadict['class_list'][i])
            for feat_name in datadict['featname_longfeatval_dict'].keys():
                new_featname_longfeatval_dict[feat_name].append( \
                                 datadict['featname_longfeatval_dict'][feat_name][i])

        datadict['arff_rows'] = new_arff_rows
        datadict['srcid_list'] = new_srcid_list
        datadict['class_list'] = new_class_list
        datadict['featname_longfeatval_dict'] = new_featname_longfeatval_dict


    def debug_compare_testtrain_data(self, rework_traindata_dict={}, rework_testdata_dict={}):
        """ for debug comparison of testdatadict and traindatadict from original un-refactored code
        """
        import cPickle, gzip
        fp = gzip.open('/tmp/actlearn_debug_orig','rb')
        orig_dict = cPickle.load(fp)
        fp.close()
        
        ### First construct a {srcid:index} dict for comparison, for bith orig and rework dicts
        orig_train_srcid_index = {}
        for i, srcid in enumerate(orig_dict['traindata_dict']['srcid_list']):
            orig_train_srcid_index[srcid] = i

        orig_test_srcid_index = {}
        for i, srcid in enumerate(orig_dict['testdata_dict']['srcid_list']):
            orig_test_srcid_index[srcid] = i

        rework_train_srcid_index = {}
        for i, srcid in enumerate(rework_traindata_dict['srcid_list']):
            rework_train_srcid_index[srcid] = i
        
        rework_test_srcid_index = {}
        for i, srcid in enumerate(rework_testdata_dict['srcid_list']):
            rework_test_srcid_index[srcid] = i


        ### Compare Training sets feature-values:
        if 1:
            print "Training comparison:"
            #for srcid, i_rework in rework_train_srcid_index.iteritems():
            for srcid, i_orig in orig_train_srcid_index.iteritems():
                #if orig_train_srcid_index.has_key(srcid):
                if rework_train_srcid_index.has_key(srcid):
                    pass #print srcid
                else:
                    print srcid, "NOT in other dataset!"
                    continue
                #i_orig = orig_train_srcid_index[srcid]
                i_rework = rework_train_srcid_index[srcid]
                for featname in rework_traindata_dict['featname_longfeatval_dict'].keys():
                    feat_val_diff = abs(rework_traindata_dict['featname_longfeatval_dict'][featname][i_rework]
                              - orig_dict['traindata_dict']['featname_longfeatval_dict'][featname][i_orig])
                    if feat_val_diff > 0.001:
                        print srcid, featname, feat_val_diff
                    #print srcid, featname, feat_val_diff

        ### Compare Testing sets feature-values:
        if 1:
            print "Testing comparison:"
            #for srcid, i_rework in rework_test_srcid_index.iteritems():
            for srcid, i_orig in orig_test_srcid_index.iteritems():
                #if orig_test_srcid_index.has_key(srcid):
                if rework_test_srcid_index.has_key(srcid):
                    pass #print srcid
                else:
                    print srcid, "NOT in other dataset!"
                    continue
                #i_orig = orig_test_srcid_index[srcid]
                i_rework = rework_test_srcid_index[srcid]
                for featname in rework_testdata_dict['featname_longfeatval_dict'].keys():
                    feat_val_diff = abs(rework_testdata_dict['featname_longfeatval_dict'][featname][i_rework]
                              - orig_dict['testdata_dict']['featname_longfeatval_dict'][featname][i_orig])
                    if feat_val_diff > 0.001:
                        print srcid, featname, feat_val_diff
                    #print srcid, featname, feat_val_diff
                        

        if 1:
            print "Training class comparison:"
            #for srcid, i_rework in rework_train_srcid_index.iteritems():
            for srcid, i_orig in orig_train_srcid_index.iteritems():
                #if orig_train_srcid_index.has_key(srcid):
                if rework_train_srcid_index.has_key(srcid):
                    pass #print srcid
                else:
                    print srcid, "NOT in other dataset!"
                    continue
                #i_orig = orig_train_srcid_index[srcid]
                i_rework = rework_train_srcid_index[srcid]
                if rework_traindata_dict['class_list'][i_rework] != orig_dict['traindata_dict']['class_list'][i_orig]:
                    print 'MISMATCH', srcid, 'ORIG:', orig_dict['traindata_dict']['class_list'][i_orig], 'REWORK:', rework_traindata_dict['class_list'][i_rework]
                print 'OK:     ', srcid, 'ORIG:', orig_dict['traindata_dict']['class_list'][i_orig], 'REWORK:', rework_traindata_dict['class_list'][i_rework]

        import pdb; pdb.set_trace()
        print
        

    def debug_compare_testtrain_data__old(self, rework_traindata_dict={}, rework_testdata_dict={}):
        """ for debug comparison of testdatadict and traindatadict from original un-refactored code
        """
        import cPickle, gzip
        fp = gzip.open('/tmp/actlearn_debug_orig','rb')
        orig_dict = cPickle.load(fp)
        fp.close()
        
        ### First construct a {srcid:index} dict for comparison, for bith orig and rework dicts
        orig_train_srcid_index = {}
        for i, srcid in enumerate(orig_dict['traindata_dict']['srcid_list']):
            orig_train_srcid_index[srcid] = i

        orig_test_srcid_index = {}
        for i, srcid in enumerate(orig_dict['testdata_dict']['srcid_list']):
            orig_test_srcid_index[srcid] = i

        rework_train_srcid_index = {}
        for i, srcid in enumerate(rework_traindata_dict['srcid_list']):
            rework_train_srcid_index[srcid] = i
        
        rework_test_srcid_index = {}
        for i, srcid in enumerate(rework_testdata_dict['srcid_list']):
            rework_test_srcid_index[srcid] = i


        ### Compare Training sets feature-values:
        if 0:
            print "Training comparison:"
            for srcid, i_rework in rework_train_srcid_index.iteritems():
                if orig_train_srcid_index.has_key(srcid):
                    print srcid
                else:
                    print srcid, "NOT in origional training set!"
                    continue
                i_orig = orig_train_srcid_index[srcid]
                for featname in rework_traindata_dict['featname_longfeatval_dict'].keys():
                    feat_val_diff = abs(rework_traindata_dict['featname_longfeatval_dict'][featname][i_rework]
                              - orig_dict['traindata_dict']['featname_longfeatval_dict'][featname][i_orig])
                    if feat_val_diff > 0.001:
                        print srcid, featname, feat_val_diff
                    print srcid, featname, feat_val_diff

        ### Compare Testing sets feature-values:
        if 0:
            print "Testing comparison:"
            for srcid, i_rework in rework_test_srcid_index.iteritems():
                if orig_test_srcid_index.has_key(srcid):
                    print srcid
                else:
                    print srcid, "NOT in origional testing set!"
                    continue
                i_orig = orig_test_srcid_index[srcid]
                for featname in rework_testdata_dict['featname_longfeatval_dict'].keys():
                    feat_val_diff = abs(rework_testdata_dict['featname_longfeatval_dict'][featname][i_rework]
                              - orig_dict['testdata_dict']['featname_longfeatval_dict'][featname][i_orig])
                    if feat_val_diff > 0.001:
                        print srcid, featname, feat_val_diff
                    #print srcid, featname, feat_val_diff
                        

        if 0:
            print "Training class comparison:"
            for srcid, i_rework in rework_train_srcid_index.iteritems():
                if orig_train_srcid_index.has_key(srcid):
                    print srcid
                else:
                    print srcid, "NOT in origional training set!"
                    continue
                i_orig = orig_train_srcid_index[srcid]
                if rework_traindata_dict['class_list'][i_rework] != orig_dict['traindata_dict']['class_list'][i_orig]:
                    print 'MISMATCH', srcid, 'ORIG:', orig_dict['traindata_dict']['class_list'][i_orig], 'REWORK:', rework_traindata_dict['class_list'][i_rework]
                print 'OK:     ', srcid, 'ORIG:', orig_dict['traindata_dict']['class_list'][i_orig], 'REWORK:', rework_traindata_dict['class_list'][i_rework]

        import pdb; pdb.set_trace()
        print


    def active_learn_main(self, session_id=0,
                                iteration_id=0,
                          session_iter_comment="",
                          n_test_subsample=100,
                          n_parts_to_divide_testset=5,
                          num_srcs_for_users=100,
                          random_seed=0):
        """ Main function which deals with running the Active Learning algorithms.

        #- SELECT a list of recently user-classified sources (to be added to trainset)
        #   - retrieve these user-classes from table: activelearn_user_class
        - generate a single list of classifications from these user classifications
           - TODO: maybe put in a seperate table of ASAS sources which we have agreed upon avg user
                   classifications.  (or just an additional column in activelearn_algo_class)
        - form the test set:
           - test ASAS sources minus user confirmed classified sources
           - ??? store this list in .arff or table?
                 - arff is probably more nondestructable
        - form the train set:
           - debosscher sources plus confirmed user classified sources
           - ??? store this list in .arff or table?
                 - arff is probably more nondestructable
        - call:  (TODO write this)
        
      new_al_dataset_for_users = \
              rpy2_classifiers.py::active_learn_randomforest(trainset, testset)
        """
        do_ignore_NA_features = False # This is a strange option to set True, which would essentially skip a featrue from being used.  The hardcoded-exclusion features are sdss and ws_ related.

        algo_code_dirpath = os.path.abspath(os.environ.get("TCP_DIR")+'Algorithms')
        sys.path.append(algo_code_dirpath)
        import rpy2_classifiers

        self.rc = rpy2_classifiers.Rpy2Classifier(algorithms_dirpath=algo_code_dirpath)

        skip_missingval_lines = False # we skip sources which have missing values
        #### These missing-value arff are for generating imputed arff files:
        #orig_arff_datadicts = self.get_orig_arff_datadicts( \
        #                               orig_trainset_arff_fpath= \
        #"/global/home/users/dstarr/scratch/arff_non_imputed_colors/tutor_123_source_feats.arff",
        #                               orig_testset_arff_fpath= \
        #"/global/home/users/dstarr/scratch/arff_non_imputed_colors/tutor_126_source_feats.arff",
        #                               skip_missingval_lines=skip_missingval_lines)

        orig_arff_datadicts = self.get_orig_arff_datadicts( \
                                       orig_trainset_arff_fpath= \
        "/home/pteluser/scratch/active_learn/imputed_arffs/train_20110805_10ntree_5mtry.arff",
                                       orig_testset_arff_fpath= \
        "/home/pteluser/scratch/active_learn/imputed_arffs/test_20110805_10ntree_5mtry.arff",
                                       skip_missingval_lines=skip_missingval_lines)

        ### with 5 color_diffs, and extinction:
        #                               orig_trainset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-07-12_23:03:43.667594/source_feats.arff",
        #                               orig_testset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-07-10_19:38:22.278633/source_feats.arff",

        ### with 2 color_diffs, and extinction:
        #                               orig_trainset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-07-14_18:06:06.233714/source_feats.arff",
        #                               orig_testset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-07-14_20:07:09.452129/source_feats.arff",
        #                               skip_missingval_lines=skip_missingval_lines)
        
        ### Used up to session=0, iter=10:
        #                               orig_trainset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-05-13_04:22:08.073940/source_feats.arff",
        #                               orig_testset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-05-13_07:19:29.802450/source_feats.arff",
        #Used for session=0, iter=1 (initial):  "/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-02-06_00:03:02.699641/source_feats.arff",
        ### Used for session=0, iter=3
        #                                orig_trainset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-04-30_00:32:56.250499/source_feats.arff",
        #                                orig_testset_arff_fpath= \
        #"/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-04-30_02:53:31.959591/source_feats.arff",

        ### Used for session=0, iter=2:
        #                               orig_trainset_arff_fpath= \
        #  "/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-02-05_23:43:21.830763/source_feats.arff",
        #                               orig_testset_arff_fpath= \
        #  "/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-03-14_18:58:32.193944/source_feats.arff",

        ##### TODO: update classes in the arff (specifically debosscher proj=123, using TUTOR database query)

        if 0:
            ### Currently this writes (testing arff) files which have imputed features
            #   - TODO: this means that eventually we want to parse these imputed arff
            #           and use in succeeding functions below.

            combo_dict = self.combine_train_and_test_datasets(orig_arff_datadicts)

            self.embarrassingly_parallelized_imputation(feature_data_dict=combo_dict['featname_longfeatval_dict'],
                                         srcid_list=combo_dict['srcid_list'],
                                         class_list=combo_dict['class_list'],
                                         arff_str=orig_arff_datadicts['test_arff_str'],
                                         train_srcids=orig_arff_datadicts['traindata_dict']['srcid_list'])
        if 0:
            ### Once the embarrassingly_parallelized_imputation() has been run and
            ###    imputed test.arff, train.arff are available, then we generate
            ###    crossvalidated errors for these .arff:
            self.get_crossvalid_errors_for_arffs()#imputed_arff_dirpath='/home/pteluser/scratch/active_learn/imputed_arffs)


        ########
        self.update_training_arffstr_classes(arff_datadicts=orig_arff_datadicts)
        

        if 0:
            ## ## ## TODO: need test &  train ['featname_longfeatval_dict'] filled for this:
            #          - in other words, need ['featname_longfeatval_dict'] extracted eary on, and then the above functions just append to this source-incremented-list if needed.
            self.apply_imputation_to_train_and_test_data(traindata_dict=orig_arff_datadicts['traindata_dict'],
                                                     testdata_dict=orig_arff_datadicts['testdata_dict'],
                                                     mtry=5, ntrees=500, nodesize=5)

        # Do cost function stuff, such as removing sources with 1day period alias:
        self.reduce_testset_using_costs(datadicts=orig_arff_datadicts)


        ### NOTE: Joey generates a vector of whether the source was in the previous AL step
        #   - also has a fector of whether both users classified the source the same
        #  TODO I will want these lists returned so they can later be used to generate cost
        #  - the training data will need all AL sources, but additionally needs the new ones identified.
        

        new_arffstrs_dict = self.incorporate_user_classified_sources_to_arffstrs( \
                                 datadicts=orig_arff_datadicts,
                                 session_id=session_id,
                                 iteration_id=iteration_id,
                                 n_test_subsample=n_test_subsample)

        if 0:
            ### For initial NRMSD vs ntree analysis:
            self.rc.test_missForest_impuation_error(orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'])


        # # # # # #
        # TODO: need to do the following in an iterative way, with 10000 sources for every iteration:
        #  - pass in the index 0, 10000 range
        #  - the following should be done in a loop:

        ### TODO: randomize the rows in the testset (NOTE: user classified sources are in trainset only)
        #random.shuffle(new_arffstrs_dict['new_testset_rows'])

        
        
        self.shuffle_srcid_rows(datadict=orig_arff_datadicts['testdata_dict'])
        
        

        (both_user_match_srcid_bool,
         actlearn_sources_freqsignifs) = self.get_cost_freqsignif_info( \
                                                srcid_list=new_arffstrs_dict['final_user_classifs']['src_id'],
                                                traindata_dict=orig_arff_datadicts['traindata_dict'],
                                                testdata_dict=orig_arff_datadicts['testdata_dict'],
                                                all_srcids_for_actlearn_iter=new_arffstrs_dict['all_srcids_for_actlearn_iter'])

        # NOTE: one extra source is retrieved for each ActiveLearning classification iteration, so that
        #       this can be added to an additional list which will have the remaining user sources randomly taken from
        #     - this means there will be at least sub_num_srcs_for_users sources used from each subgroup iteration
        sub_num_srcs_for_users = 1 + (num_srcs_for_users - len(self.pars['high_conf_srcids'])) / n_parts_to_divide_testset

        combo_result_dict = {'actlearn_tups':[],
                             'srcid_list':[],
                             'err.decr':[], # index must match testdata_dict['srcid_list']
                             'new_testset_rows':orig_arff_datadicts['testdata_dict']['arff_rows'], # should be the all/combo dict
                             'select_rows':[],
                             'allsrc_tups':[],
                             'all.pred':[],
                             'all.predprob':[],
                             'all_top_prob':[]}

        remainder_result_dict = {'actlearn_tups':[],
                             'srcid_list':[],
                             'err.decr':[], # index must match testdata_dict['srcid_list']
                             'select_rows':[],
                             'allsrc_tups':[],
                             'all.pred':[],
                             'all.predprob':[],
                             'all_top_prob':[]}
        i_range_tups = []
        #for i in [0,1]:
        for i in range(n_parts_to_divide_testset):
            i_low =  (i)   * (len(orig_arff_datadicts['testdata_dict']['arff_rows']) / n_parts_to_divide_testset)
            i_high = (i+1) * (len(orig_arff_datadicts['testdata_dict']['arff_rows']) / n_parts_to_divide_testset)
            print i, '/', n_parts_to_divide_testset, i_low, i_high

            testset_rows_subset = orig_arff_datadicts['testdata_dict']['arff_rows'][i_low:i_high] # TODO: make sure no overlap

            testdata_dict = {'class_list':orig_arff_datadicts['testdata_dict']['class_list'][i_low:i_high],
                             'srcid_list':orig_arff_datadicts['testdata_dict']['srcid_list'][i_low:i_high],
                             'arff_rows':orig_arff_datadicts['testdata_dict']['arff_rows'][i_low:i_high],
                             'featname_longfeatval_dict':{},
                             }
            for feat_name in orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
                testdata_dict['featname_longfeatval_dict'][feat_name] = []
            for feat_name in orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'].keys():
                for k in range(i_low, i_high):
                    testdata_dict['featname_longfeatval_dict'][feat_name].append( \
                                  orig_arff_datadicts['testdata_dict']['featname_longfeatval_dict'][feat_name][k])

            if 0:
                # KLUDGE: 
                if '@data' in new_arffstrs_dict['test_arff_str'][:50000]:
                    new_test_arff_str = new_arffstrs_dict['test_arff_str'][: \
                                                               new_arffstrs_dict['test_arff_str'].find('@data')+5]
                else:
                    new_test_arff_str = new_arffstrs_dict['test_arff_str'][: \
                                                               new_arffstrs_dict['test_arff_str'].find('@DATA')+5]
                testset_rows_subset.append('\n')
                new_test_arff_str += '\n'                
                new_test_arff_str += '\n'.join(testset_rows_subset)
                ## ## ## TODO: just re-form the train_arff_str here (or used a reformed one outside of the i-loop)
                sub_arffstrs_dict = {'train_arff_str':new_arffstrs_dict['train_arff_str'],
                                     'test_arff_str':new_test_arff_str,
                                     'new_testset_rows':testset_rows_subset,
                                     'srcids_to_addto_trainarffstr':new_arffstrs_dict['srcids_to_addto_trainarffstr']}
                # ??? has 'srcids_to_addto_trainarffstr' changed?

                ### Just for archiving / debugging use:
                sess_iter_str = "sess%d_iter%d" % (session_id, iteration_id)
                fp = open('/home/pteluser/scratch/active_learn/actlearn_%s_train_%d.arff' % (sess_iter_str, i), 'w')
                #fp.write(new_arffstrs_dict['train_arff_str'])
                fp.write(sub_arffstrs_dict['train_arff_str'])
                fp.close()
                fp = open('/home/pteluser/scratch/active_learn/actlearn_%s_test_%d.arff' % (sess_iter_str, i), 'w')
                #fp.write(new_arffstrs_dict['test_arff_str'])
                fp.write(sub_arffstrs_dict['test_arff_str'])
                fp.close()

                traindata_dict = self.rc.parse_full_arff(arff_str=sub_arffstrs_dict['train_arff_str'], skip_missingval_lines=skip_missingval_lines)

                testdata_dict = self.rc.parse_full_arff(arff_str=sub_arffstrs_dict['test_arff_str'], skip_missingval_lines=skip_missingval_lines)


            ### Need to identify which sources in the traindata_dict are derived from ActLearn user classifications
            #    - KLUDGY: Maybe do in R:

            # This is just uglier to work with:
            #actlearn_used_srcids_indicies = [traindata_dict['srcid_list'].index(str(srci)) for srci in new_arffstrs_dict['final_user_classifs']['src_id']]
            #import pdb; pdb.set_trace()
            #print


            ## ## ## ## ## ## ## ## ## ## ##


            ### Not used due to useing partitioned test set and thus we hardcoded
            ### rpye_classifiers.py::actlearn_randomforest()::getCost()::sig.vec = 0:36
            #actlearn_used_srcids_indicies = []
            #both_user_match_srcid_bool = []
            #for srci in new_arffstrs_dict['final_user_classifs']['src_id']:
            #    if str(srci) in testdata_dict['srcid_list']:
            #        actlearn_used_srcids_indicies.append(testdata_dict['srcid_list'].index(str(srci)))
            #        if srci in new_arffstrs_dict['final_user_classifs']['both_user_match_dict']['src_id']:
            #            both_user_match_srcid_bool.append(True)
            #        else:
            #            both_user_match_srcid_bool.append(False)
            #        



            traindata_dict = orig_arff_datadicts['traindata_dict']


            if 0:
                #for debug comparison of testdatadict and traindatadict from original un-refactored code
                self.debug_compare_testtrain_data(rework_traindata_dict=traindata_dict,
                                                  rework_testdata_dict=testdata_dict)

            if 0:
                import cPickle, gzip
                fp = gzip.open('/tmp/actlearn_debug_rework','wb')
                debug_dict = {'traindata_dict':traindata_dict,
                              'testdata_dict':testdata_dict}
                cPickle.dump(debug_dict,fp,1) # ,1) means a binary pkl is used.
                fp.close()
                import pdb; pdb.set_trace()
                print


            #############################
            if 1:
                
                traindata_dict = eval(pprint.pformat(traindata_dict))
                testdata_dict = eval(pprint.pformat(testdata_dict))
            
            ##############################

            classifier_dict = self.rc.actlearn_randomforest(traindata_dict=traindata_dict,
                                                            testdata_dict=testdata_dict,
                                                            do_ignore_NA_features=do_ignore_NA_features,
                                                            mtry=5, ntrees=500, nodesize=5,
                                                            num_srcs_for_users=sub_num_srcs_for_users,
                                                            random_seed=random_seed,
                                                            both_user_match_srcid_bool=both_user_match_srcid_bool,
                                                            actlearn_sources_freqsignifs=actlearn_sources_freqsignifs,
                                                            )
            #                                          actlearn_used_srcids_indicies=actlearn_used_srcids_indicies,
            #                                                both_user_match_srcid_bool=both_user_match_srcid_bool)

            # TODO: we actually want to skip the last source in each list
            temp_select_rows = [testdata_dict['arff_rows'][j] for j in classifier_dict['select']] # 3
            combo_result_dict['actlearn_tups'].extend(classifier_dict['actlearn_tups'][:-1]) # 3
            combo_result_dict['srcid_list'].extend(testdata_dict['srcid_list'][:-1]) # 1508 (45k/30)
            combo_result_dict['err.decr'].extend(classifier_dict['err.decr'][:-1]) # 1508 index must match testdata_dict['srcid_list']
            #combo_result_dict['new_testset_rows'].extend(new_arffstrs_dict['new_testset_rows'][:-1]) #45298 should be the all/combo dict
            combo_result_dict['select_rows'].extend(temp_select_rows[:-1])
            combo_result_dict['allsrc_tups'].extend(classifier_dict['allsrc_tups'][:-1]) # 3

            combo_result_dict['all.pred'].extend(classifier_dict['all.pred'][:-1])
            combo_result_dict['all.predprob'].extend(classifier_dict['all.predprob'][:-1])
            combo_result_dict['all_top_prob'].extend(classifier_dict['all_top_prob'][:-1])


            ##################################################
            remainder_result_dict['actlearn_tups'].append(classifier_dict['actlearn_tups'][-1]) # 3
            remainder_result_dict['srcid_list'].append(testdata_dict['srcid_list'][-1]) # 1508 (45k/30)
            remainder_result_dict['err.decr'].append(classifier_dict['err.decr'][-1]) # 1508 index must match testdata_dict['srcid_list']
            #remainder_result_dict['new_testset_rows'].append(new_arffstrs_dict['new_testset_rows'][-1]) #45298 should be the all/combo dict
            remainder_result_dict['select_rows'].append(temp_select_rows[-1])
            remainder_result_dict['allsrc_tups'].append(classifier_dict['allsrc_tups'][-1]) # 3

            remainder_result_dict['all.pred'].append(classifier_dict['all.pred'][-1]) # 3
            remainder_result_dict['all.predprob'].append(classifier_dict['all.predprob'][-1]) # 3
            remainder_result_dict['all_top_prob'].append(classifier_dict['all_top_prob'][-1]) # 3


            ##################################################


            if 0:
                for a,b in traindata_dict.iteritems():
                    print 'iter=', i, 'traindata_dict', a, len(b),
                    if len(b) > 0:
                        try:
                            print b[0]
                        except:
                            print
                    else:
                        print 
                    
                for a,b in testdata_dict.iteritems():
                    print 'iter=', i, 'testdata_dict', a, len(b),
                    if len(b) > 0:
                        try:
                            print b[0]
                        except:
                            print
                    else:
                        print 

                for a,b in combo_result_dict.iteritems():
                    print 'iter=', i, 'combo_result_dict', a, len(b),
                    if len(b) > 0:
                        try:
                            print b[0]
                        except:
                            print
                    else:
                        print 

                for a,b in remainder_result_dict.iteritems():
                    print 'iter=', i, 'remainder_result_dict', a, len(b),
                    if len(b) > 0:
                        print b[0]
                    else:
                        print 


        ### Add additional high confidence sources to the list of active learning sources,
        ###    - which will be added to the database
        self.add_remainder_to_combo(combo_result_dict=combo_result_dict,
                                    remainder_result_dict=remainder_result_dict,
                                    num_srcs_for_users=num_srcs_for_users)


        self.add_highconfidence_sources_to_combo(combo_result_dict=combo_result_dict,
                                                 remainder_result_dict=remainder_result_dict,
                                                 num_srcs_for_users=num_srcs_for_users)

        import pdb; pdb.set_trace()
        print

        ### This stores a copy of a PvsAmp plot of Active Learned sources on top of Debosscher/training soures:
        self.analysis_plot_period_amp(test_arff_rows=orig_arff_datadicts['testdata_dict']['arff_rows'],#50115
                                      train_arff_rows=orig_arff_datadicts['traindata_dict']['arff_rows'],#1542
                                      new_testset_rows=combo_result_dict['new_testset_rows'],#1000
                                      test_srcid_list=combo_result_dict['srcid_list'],
                                      select_rows=combo_result_dict['select_rows'],
                                      session_id=session_id,
                                      iteration_id=iteration_id,
                                      n_test_subsample=n_test_subsample)
        # {'py_obj':classifier_out,
        #        'r_name':'rf_clfr',
        #        'select':robjects.globalenv['select'],
        #        'select.pred':robjects.r("rf_clfr$test$predicted[select]"),
        #        'select.predprob':robjects.r("rf_clfr$test$votes[select,]"),
        #        'err.decr':robjects.globalenv['err.decr'],
        #        'all.pred':robjects.r("rf_clfr$test$predicted"),
        #        'all.predprob':robjects.r("rf_clfr$test$votes"),
        #        }

        rclass_tutorid_lookup = self.retrieve_tutor_class_ids()

        #import pdb; pdb.set_trace()
        #print
        

        #print classifier_dict.keys()
        #print classifier_dict['select.predprob']

        # (src_id, rank, prob, deb_class) = tup
        #   int     int  flt    R_class

        ### act_iter_id += 1 ??? 
        #import pdb; pdb.set_trace()
        #print

        self.insert_tups_into_rdb(rclass_tutorid_lookup=rclass_tutorid_lookup,
                                  actlearn_tups=combo_result_dict['actlearn_tups'],
                                  tup_list=combo_result_dict['allsrc_tups'],
                                  act_session_id=session_id,
                                  act_iter_id=iteration_id,
                                  n_user_classifs=len(new_arffstrs_dict['srcids_to_addto_trainarffstr']),
                                  error_rate=1.0,  # initally this error rate is not useful since it is in relation with the ASAS classifications which are incomplete.
                                  session_iter_comment=session_iter_comment)


        ################
        if 0:
            ##### TESTING: This is just used to print the lowest Active Learning Importance, which
            #    are often confidently sampled and classified sources:
            all_tups_list = []
            i = -1
            for (srcid, rank, prob, class_name) in classifier_dict['allsrc_tups']:
                if rank == 0:
                    i += 1
                    importance = list(classifier_dict['err.decr'])[i]
                    all_tups_list.append((importance, srcid, prob, class_name))
            all_tups_list.sort()
            for (importance, srcid, prob, class_name) in all_tups_list[:100]:
                print importance, srcid, prob, class_name
        ################


        #classes = [classifier_dict['possible_classes'][i] for i in list(classifier_dict['all.pred'])]
        #x = zip(list(classifier_dict['err.decr']), testdata_dict['srcid_list'], classes)
        #x.sort()
        #for (importance, srcid, class_name) in x[:1000]:
        #    print importance, srcid, class_name#

        #classifier_dict['possible_classes']
        import pdb; pdb.set_trace()
        print


    def fill_user_leaderboard_tables(self, session_id=0,
                                     iteration_id=0):
        """
        NOTE: Do this for existing, user-completed iterations

        For AL (and high confidence) sources from last iteration:
         - get current classifiers classifications for these sources
         - select the last classification for each user, for these sources,
         - calculate metrics for each user:
            - (user_id, session_id, iter_id)
            - percent classified with some science class
            - percent (of those classified) which match group consensus
            - percent (of those classified) which match current AL classifier
            - for each sciclass, percent correct when matching group consensus
        """

        ### Get act_id
        select_str = "SELECT act_id FROM activelearn_session WHERE act_session=%d AND act_iter=%d" % ( \
                      session_id, iteration_id)
        self.tcp_cursor.execute(select_str)
        results = self.tcp_cursor.fetchall()
        if len(results) == 0:
            raise "ERROR"
        act_id = int(results[0][0])

        ### Get user_id which made classifications for this iteration
        select_str = """
SELECT activelearn_user_class.user_id, count(activelearn_user_class.source_id)
FROM activelearn_srcid_importance
JOIN activelearn_user_class ON (activelearn_user_class.source_id=activelearn_srcid_importance.source_id)

WHERE activelearn_srcid_importance.act_id=%d
GROUP BY user_id
HAVING  count(activelearn_user_class.source_id) > 0
        """ % (act_id)

        self.tcp_cursor.execute(select_str)
        results = self.tcp_cursor.fetchall()
        if len(results) == 0:
            raise "ERROR"
        user_ids = []
        for user_id, count in results:
            print user_id, count
            user_ids.append(int(user_id))
            

        user_totals = {}

        ### Get | source_id | user_class | al_class | class_diff |
        ###     for a user_id, act_id
        for user_id in user_ids:
            select_str = """
SELECT activelearn_srcid_importance.source_id,
       activelearn_user_class.tutor_class_id AS user_class,
       activelearn_algo_class.tutor_class_id AS al_class,
       activelearn_algo_class.tutor_class_id - activelearn_user_class.tutor_class_id AS class_diff
FROM activelearn_srcid_importance
JOIN activelearn_user_class ON (activelearn_user_class.source_id=activelearn_srcid_importance.source_id
                                AND activelearn_user_class.user_id=%d)
JOIN activelearn_algo_class ON (activelearn_algo_class.source_id=activelearn_srcid_importance.source_id 
                                AND activelearn_algo_class.act_id=%d
                                AND activelearn_algo_class.rank=0)
WHERE activelearn_srcid_importance.act_id=%d
            """ % (user_id, act_id, act_id)

            self.tcp_cursor.execute(select_str)
            results = self.tcp_cursor.fetchall()
            if len(results) == 0:
                raise "ERROR"

            user_totals[user_id] = {'al_match_count':0, # num of userclassifs which match Actlearn
                                    'group_match_count':0, # num of userclassifs which match group consensus class
                                    'n_sciclassifs':0,  # num of sources the user gave science classifs for
                                    'n_sources':len(results),
                                    'n_group_srcs':len(self.user_classifs[session_id][iteration_id]['src_id']),
                                    'class_match_dict':{}, # num of sources, for a class which a user correctly
                                    'class_total_dict':{}} # total num of sources, for a class

            for src_id__long, user_class, al_class, class_diff in results:
                src_id = int(src_id__long)
                print user_id, src_id, user_class, al_class, class_diff
                if class_diff != None:
                    user_totals[user_id]['n_sciclassifs'] += 1
                    #if not user_totals[user_id]['class_total_dict'].has_key(al_class):
                    #    user_totals[user_id]['class_total_dict'][al_class] = 0
                    #    user_totals[user_id]['class_match_dict'][al_class] = 0
                    #user_totals[user_id]['class_total_dict'][al_class] += 1  # when counting against AL classifs
                    if class_diff == 0:
                        user_totals[user_id]['al_match_count'] += 1
                        #user_totals[user_id]['class_match_dict'][al_class] += 1 # when counting against AL classifs
                    if self.user_classifs[session_id][iteration_id]['srcid_class_dict'].has_key(src_id):
                        ### Then this src_id had an agreed-upon consensus classification:
                        group_class = self.user_classifs[session_id][iteration_id]['srcid_class_dict'][src_id]
                        if not user_totals[user_id]['class_total_dict'].has_key(group_class):
                            user_totals[user_id]['class_total_dict'][group_class] = 0
                            user_totals[user_id]['class_match_dict'][group_class] = 0
                        user_totals[user_id]['class_total_dict'][group_class] += 1
                        if user_class == group_class:
                            user_totals[user_id]['group_match_count'] += 1
                            user_totals[user_id]['class_match_dict'][group_class] += 1


        ### insert Overall user percentages into table:
        insert_list = ["INSERT INTO activelearn_leaderboard (act_session, act_iter, user_id, perc_sciclass, perc_match_al, perc_match_group) VALUES "]
        for user_id in user_totals.keys():
            insert_list.append("(%d, %d, %d, %lf, %lf, %lf), " % ( \
                               session_id, iteration_id, user_id,
                               user_totals[user_id]['n_sciclassifs']/float(user_totals[user_id]['n_sources']),
                               user_totals[user_id]['al_match_count']/float(user_totals[user_id]['n_sources']),
                               user_totals[user_id]['group_match_count']/float(user_totals[user_id]['n_group_srcs']),
                ))

        insert_str = ''.join(insert_list)[:-2] + """ ON DUPLICATE KEY UPDATE
        perc_sciclass=VALUES(perc_sciclass),
        perc_match_al=VALUES(perc_match_al),
        perc_match_group=VALUES(perc_match_group)
        """
        self.tcp_cursor.execute(insert_str)


        ### insert by-class, act_id percentages into table:
        insert_list = ["INSERT INTO activelearn_user_class_stats (act_session, act_iter, user_id, tutor_class_id, n_correct, n_total) VALUES "]
        for user_id in user_totals.keys():
            for class_id in user_totals[user_id]['class_total_dict'].keys():
                insert_list.append("(%d, %d, %d, %d, %lf, %lf), " % ( \
                               session_id, iteration_id, user_id,
                               class_id,
                               user_totals[user_id]['class_match_dict'][class_id],
                               user_totals[user_id]['class_total_dict'][class_id],
                ))

        insert_str = ''.join(insert_list)[:-2] + """ ON DUPLICATE KEY UPDATE
        n_correct=VALUES(n_correct),
        n_total=VALUES(n_total)
        """
        self.tcp_cursor.execute(insert_str)






if __name__ == '__main__':

    ### NOTE: most of the RDB parameters were dupliclated from ingest_toolspy::pars{}
    pars = { \
    'tutor_hostname':'192.168.1.103',
    'tutor_username':'dstarr', #'tutor', # guest
    'tutor_password':'ilove2mass', #'iamaguest',
    'tutor_database':'tutor',
    'tutor_port':3306, #33306,
    'tcp_hostname':'192.168.1.25',
    'tcp_username':'pteluser',
    'tcp_port':     3306, #23306, 
    'tcp_database':'source_test_db',
    'asas_duplicate_sources':['225529', '226314', '226371', '226397', '226488', '226501', '227554', '231054', '215322', '216865', '235114', '239641', '240672', '241340', '242867', '243391', '244155', '244331', '218112', '246414', '246642', '247931', '248144', '248216', '248532', '248723', '249009', '249481', '249594', '249759', '249777', '249823', '249869', '250375', '250499', '250686', '251277', '218795', '218815', '252521', '253530', '254491', '254614', '219106', '255145', '256456', '260272', '219713', '219721', '219744', '261096', '261123', '219756', '261202', '219796', '262618', '262703', '220060', '264293', '264720', '220148', '220678', '220811', '220975', '215756', '221975', '222493', '222854', '222995', '215950', '215971', '223547', '223658', '224378', '224438', '224619', '224623', '225144'],
    'high_conf_srcids':range(12), # range(22) #[241682, 238040, 221547, 225633, 227203, 250761, 219325, 252782, 245584, 236706, 216173, 225396, 233750, 232693, 263653, 216768, 225919, 264626, 230520, 229680, 231266, 221448, 226872, 261712], # Used for session=0, iter=1 (1st)
    'user_classifs_fpaths':{ \
        1:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_1.dat'), #04/15/2011 AL_addtotrain_1.dat >= 2011-04-08 22:34:01
        2:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_2.dat'), #05/02/2011 AL_addToTrain_2.dat >= 2011-04-21 12:44:19
        3:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_3.dat'), #05/11/2011 AL_addToTrain_3.dat >= 2011-05-04 12:13:16
        4:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_4.dat'), #05/21/2011 AL_addToTrain_4.dat >= 2011-05-13 00:48:27
        5:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_5.dat'), #05/23/2011 AL_addToTrain_5.dat >= 2011-05-21 19:11:09
        6:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_6.dat'), #05/25/2011 AL_addToTrain_6.dat >= 2011-05-24 14:37:51
        7:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_7.dat'), #05/26/2011 AL_addToTrain_7.dat >= 2011-05-25 16:50:28
        8:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_8.dat'), #05/27/2011 AL_addToTrain_8.dat >= 2011-05-27 10:50:00
        9:os.path.expandvars('$TCP_DIR/Data/allstars/AL_addToTrain_9.dat'), # >= 2011-05-27 12:05:57
        10:os.path.expandvars('$TCP_DIR/Data/allstars/AL_SIMBAD_confirmed.dat'), # SIMBAD confirmed sources
        }, # NOTE: The above files are now algo generated, with 5 sources added, per 20110526 discussion with Joey.
    'user_classifs_joey_algo_chosen':{ \
        1:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_1.dat'),
        2:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_2.dat'),
        3:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_3.dat'),
        4:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_4.dat'),
        5:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_5.dat'),
        6:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_6.dat'),
        7:os.path.expandvars('$HOME/scratch/actlearn_joey_algo_chosen_userclassifs/AL_addToTrain_7.dat'),
        },
    }

    if 0:
        ### Do a data-cleaning check of sources in AL_SIMBAD_confirmed.dat file which
        ###  may be be repeats in earlier AL_addToTrain_?.dat files.

        check_userAL_overlap_with_simbad_matched_sources(pars)
        sys.exit()

    ### for debugging AL_addToTrain_*.dat lists (original and crowdsoruceing algorithm based):
    #compare_userclassifs_files(pars=pars)


    #DatabaseUtils = Database_Utils(pars=pars)
    #DatabaseUtils.create_tables() # Do this one time only.
    #DatabaseUtils.add_data_into_new_tables() # Do this one time only.
    #sys.exit()


    #MANUAL#ActiveLearn.fill_user_leaderboard_tables(session_id=0,
    #                                                iteration_id=1) # Do this for user-completed iterations ( -1 from active_learn_main(iteration_id) 


    #train_arff_str = open(os.path.expandvars("/media/raid_0/historical_archive_featurexmls_arffs/tutor_123/2011-02-05_23:43:21.830763/source_feats.arff")).read()
    #test_arff_str = open(os.path.expandvars("/media/raid_0/historical_archive_featurexmls_arffs/tutor_126/2011-02-06_00:03:02.699641/source_feats.arff")).read()
    #OLDER#ActiveLearn.generate_initial_classifications(train_arff_str=train_arff_str, test_arff_str=test_arff_str)
    if 0:
        ######################## DEBUG / stripe82 INSERT into tutor121_algo_class
        ActiveLearn = Active_Learn(pars=pars)
        iteration_id=10 # This is +1 more than the last finished iteration
        ActiveLearn.retrieve_user_consensus_classifications(iteration_id=iteration_id,
                                                            session_id=0)
        import pdb; pdb.set_trace()

        ActiveLearn.apply_actlearn_classifier_to_arff(session_id=0,
                                      iteration_id=iteration_id,
                                      session_iter_comment="",
                                      n_test_subsample=0, # (becoming obsolete) 20000 max?
                                      n_parts_to_divide_testset=4, #4, # 3=roughly 16700 sources per division
                                      num_srcs_for_users=62,# (assuming 50AL and 10highconf)  # NOTE: If we use 100 here and this will have the 25 high-confids automatically included, so the AL generated will be (100-25) = 75  when num_srcs_for_users==100
                                      random_seed=1234)
        sys.exit()

    ########################
    # Current:
    ### iteration_id=1, # Prior to 2011-04-12 (first AL dataset given)
    iteration_id=11 # currently finishing 9
    ActiveLearn = Active_Learn(pars=pars)
    ActiveLearn.retrieve_user_consensus_classifications(iteration_id=iteration_id,
                                                        session_id=0)
    ActiveLearn.active_learn_main(session_id=0,
                                  iteration_id=iteration_id,
                                  session_iter_comment="Adding SIMBAD/ASAS which miller, starr confirmed",
                                  n_test_subsample=0, # (becoming obsolete) 20000 max?
                                  n_parts_to_divide_testset=2, #4, # 3=roughly 16700 sources per division
                                  num_srcs_for_users=62,# (assuming 50AL and 10highconf)  # NOTE: If we use 100 here and this will have the 25 high-confids automatically included, so the AL generated will be (100-25) = 75  when num_srcs_for_users==100
                                  random_seed=1234)

    ActiveLearn.fill_user_leaderboard_tables(session_id=0,
                                             iteration_id=iteration_id - 1) # Do this for user-completed iterations ( -1 from active_learn_main(iteration_id) 
