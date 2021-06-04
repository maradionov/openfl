from concurrent.futures import ProcessPoolExecutor
import logging
from grpc import aio
import asyncio
from pathlib import Path
from collections import defaultdict

from openfl.protocols import preparations_pb2
from openfl.protocols import preparations_pb2_grpc
from openfl.federated import Plan

logger = logging.getLogger(__name__)


class Director(preparations_pb2_grpc.FederationDirectorServicer):

    def __init__(self, sample_shape, target_shape) -> None:
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self.shard_registry = []
        self.experiments = set()
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.experiment_data = {}
        self.experiments_queue = asyncio.Queue()
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.aggregator_task = None  # TODO: add check if exists and wait on terminate

    async def AcknowledgeShard(self, shard_info, context):
        reply = preparations_pb2.ShardAcknowledgement(accepted=False)
        # If dataset do not match the data interface of the problem
        if (self.sample_shape != shard_info.sample_shape) or \
                (self.target_shape != shard_info.target_shape):
            return reply

        self.shard_registry.append(shard_info)
        print('\n\n\nRegistry now looks like this\n\n', self.shard_registry)
        reply.accepted = True
        return reply

    async def SetNewExperiment(self, stream, context):
        logger.info(f'SetNewExperiment request has got {stream}')
        # TODO: add streaming reader
        npbytes = b""
        async for request in stream:
            if request.experiment_data.size == len(request.experiment_data.npbytes):
                npbytes += request.experiment_data.npbytes
            else:
                raise Exception('Bad request')

        logger.info(f'New experiment {request.name} for collaborators {request.collaborator_names}')
        # TODO: save to file
        self.experiment_data[request.name] = npbytes

        # TODO: add a logic with many experiments
        for col_name in request.collaborator_names:
            queue = self.col_exp_queues[col_name]
            await queue.put(request.name)

        future = self.executor.submit(self._run_aggregator)
        self.aggregator_task = future

        return preparations_pb2.Response(accepted=True)

    async def GetExperimentData(self, request, context):
        # experiment_data = preparations_pb2.ExperimentData()
        # with open(experiment_name + '.zip', 'rb') as content_file:
        #     content = content_file.read()
        #     # TODO: add size filling
        #     # TODO: add experiment name field
        #     # TODO: rename npbytes to data
        content = self.experiment_data.get(request.experiment_name, b'')
        logger.info(f'Content length: {len(content)}')
        max_buffer_size = (2 * 1024 * 1024)

        for i in range(0, len(content), max_buffer_size):
            chunk = content[i:i + max_buffer_size]
            logger.info(f'Send {len(chunk)} bytes')
            yield preparations_pb2.ExperimentData(size=len(chunk), npbytes=chunk)

    async def WaitExperiment(self, request_iterator, context):
        logger.info('Request WaitExperiment has got!')
        async for msg in request_iterator:
            logger.info(msg)
        queue = self.col_exp_queues[msg.collaborator_name]
        experiment_name = await queue.get()
        logger.info(f'Experiment {experiment_name} was prepared')

        yield preparations_pb2.WaitExperimentResponse(experiment_name=experiment_name)

    @staticmethod
    def _run_aggregator(plan='plan/plan.yaml',
                        authorized_cols='plan/cols.yaml'):  # TODO: path params, change naming
        plan = Plan.Parse(
            plan_config_path=Path(plan),
            cols_config_path=Path(authorized_cols)
        )

        logger.info('🧿 Starting the Aggregator Service.')

        plan.get_server().serve(disable_tls=True, disable_client_auth=True)


async def serve(*args, **kwargs):
    logging.basicConfig()
    server = aio.server()
    preparations_pb2_grpc.add_FederationDirectorServicer_to_server(Director(*args, **kwargs),
                                                                   server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logger.info(f'Starting server on {listen_addr}')
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
