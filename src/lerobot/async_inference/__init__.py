# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Async inference server/client.

Server/client modules require: ``pip install 'lerobot[async]'``

The gRPC gate is enforced inside :mod:`lerobot.async_inference.policy_server` and
:mod:`lerobot.async_inference.robot_client` (the only modules that actually
import ``grpc``). Lightweight utilities in this package — ``constants``,
``configs``, ``helpers``, ``local_planner`` — can be imported without the
``async`` extra installed.

Available modules (import directly)::

    from lerobot.async_inference.policy_server import ...
    from lerobot.async_inference.robot_client import ...
    from lerobot.async_inference.local_planner import LocalAsyncPlanner
"""

__all__: list[str] = []
