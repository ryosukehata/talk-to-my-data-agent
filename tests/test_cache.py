# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from typing import Any

import pytest

from utils.schema import AnalystDataset


@pytest.mark.asyncio
async def test_get_dictionary_cache(
    pulumi_up: Any, dataset_loaded: AnalystDataset
) -> None:
    from utils.api import get_dictionaries

    # shuffle data to invalidate cache
    df_shuffled = dataset_loaded.to_df().sample(frac=1, random_state=42)
    new_dataset_loaded = AnalystDataset(name=dataset_loaded.name, data=df_shuffled)

    # First call - populate cache and measure time
    start_time = datetime.now()
    result1 = await get_dictionaries([new_dataset_loaded])
    end_time = datetime.now()
    duration1 = (end_time - start_time).total_seconds()

    print(f"First call duration: {duration1:.2f} seconds")

    # Second call - should retrieve from cache and measure time
    start_time = datetime.now()
    result2 = await get_dictionaries([new_dataset_loaded])
    end_time = datetime.now()
    duration2 = (end_time - start_time).total_seconds()

    print(f"Second call duration: {duration2:.2f} seconds")

    # Assertion to check if the second call is significantly faster
    assert result1 == result2, "Cached results must be identical"
    assert duration2 < duration1 * 0.5, (
        "Second call must be at least half as fast as the first"
    )
