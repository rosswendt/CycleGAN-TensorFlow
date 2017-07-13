"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import sys
import re

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('output_prefix', 'inference_', 'prefix appended to input image name(s) to create output image name(s), default: inference_')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_string('multi_input', None, 'input image directory for multiple image')
tf.flags.DEFINE_string('multi_output', None, 'output image directory for multiple image')
tf.flags.DEFINE_integer('batch_size', '5', 'batch size, default: 5')
tf.flags.DEFINE_string('single_input', 'input_sample.jpg', 'input image path for a single image (.jpg)')
tf.flags.DEFINE_string('single_output', 'output_sample.jpg', 'output image path for a single image (.jpg)')

def inference():
    if FLAGS.multi_input is None:
        file_names = [FLAGS.single_input]
        batched_file_names = [file_names]
        outfile_path = ""
        input_path = ""

    if FLAGS.multi_input is not None:
        file_names = [file_name for file_name in os.listdir(FLAGS.multi_input) if re.findall(".jpg", file_name)]
        batched_file_names = [file_names[x:x + FLAGS.batch_size] for x in range(0, len(file_names), FLAGS.batch_size)]
        outfile_path = FLAGS.multi_output + '/' + FLAGS.output_prefix
        input_path = FLAGS.multi_input + '/'

    output_files = [] #created, appended to, cleared, and appended to again for each batch
    progress_counter = 1
    for single_batch in batched_file_names:
        graph = tf.Graph()
        with graph.as_default():
            for filename in single_batch:
                with tf.gfile.FastGFile(input_path + filename, 'rb') as f:
                    image_data = f.read()
                    input_image = tf.image.decode_jpeg(image_data, channels=3)
                    input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
                    input_image = utils.convert2float(input_image)
                    input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

                with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(model_file.read())
                [output_image] = tf.import_graph_def(graph_def,
                                                     input_map={'input_image': input_image},
                                                     return_elements=['output_image:0'],
                                                     name='output')
                output_files.append(output_image)
                sys.stdout.write("\r{} of {} files prepared".format(progress_counter, len(file_names)))
                progress_counter+=1
            print()
        with tf.Session(graph=graph) as sess:
            print("Performing inference and writing " + str(len(output_files)) + " files to disk")
            for image, single_file in zip(output_files, single_batch):
                generated = image.eval()
                if FLAGS.multi_output is None:
                    outfile = str(FLAGS.single_output)
                else:
                    outfile = outfile_path + str(single_file)
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                with open(outfile, 'wb') as f:
                    f.write(generated)
                    f.flush()
            del output_files[:]

def main(unused_argv):
    inference()

if __name__ == '__main__':
    tf.app.run()
