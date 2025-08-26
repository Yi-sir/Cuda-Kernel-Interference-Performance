import logging

import torch
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stream_pairs(device: torch.device):
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    logger.info(f"Current device: {device}, sm count is {sm_count}")

    num_sms_start = 8
    num_sms_step = 8
    num_sms_end = (sm_count // num_sms_step) * num_sms_step
    num_sms = [i for i in range(num_sms_start, num_sms_end, num_sms_step)]
    num_sm_counts = len(num_sms)

    stream_pairs = []

    for i in range((num_sm_counts)):
        sm_count_0 = num_sms[i]
        sm_count_1 = num_sms[num_sm_counts - 1 - i]

        logger.info(f"Creating green ctx with SM counts: {sm_count_0}, {sm_count_1}")

        (stream_0, stream_1, _), resources = split_device_green_ctx_by_sm_count(
            device,
            [sm_count_0, sm_count_1],
        )

        stream_pairs.append((stream_0, stream_1, sm_count_0, sm_count_1))
        # logger.info(f"Remaining SM resources: {[r.sm.smCount for r in resources]}")

    return stream_pairs


def get_all_streams(device: torch.device):
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    logger.info(f"Current device: {device}, sm count is {sm_count}")

    num_sms_start = 8
    num_sms_step = 8
    num_sms_end = (sm_count // num_sms_step) * num_sms_step
    num_sms = [i for i in range(num_sms_start, num_sms_end, num_sms_step)]
    num_sm_counts = len(num_sms)

    streams = []

    for sms in num_sms:
        logger.info(f"Creating green ctx with SM count: {sms}")
        (stream, _), _ = split_device_green_ctx_by_sm_count(device, [sms])

        streams.append((stream, sms))

    return streams


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:7")
        streams = get_stream_pairs(device)
    else:
        logger.info("torch cuda is not available!")
