# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "generate_shapes_graph",
    srcs = ["generate_shapes_graph.py", "sketch_utils.py"],
)

py_binary(
    name = "model_exporter",
    srcs = [
        "model_exporter.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":generate_shapes_graph",
        "//tensorflow_serving/session_bundle:exporter",
    ],
)
