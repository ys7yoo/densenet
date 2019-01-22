# base class for DNNs

class BaseNet:
    def __init__(self, data_provider,
                 dataset_name,
                 should_save_logs, should_save_model):
        self.num_inter_threads = num_inter_threads
        self.num_intra_threads = num_intra_threads
        self._logs_path = logs_path
        self.saver = tf.train.Saver()
        self.dataset_name = dataset_name
        #self.sess = None

        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes

        self.model_type = model_type

        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs



    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)


    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def load(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def save(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()

        # Specify the CPU inter and Intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads

        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logs_writer = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logs_writer = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logs_writer(self.logs_path)

