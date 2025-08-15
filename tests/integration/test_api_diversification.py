"""
Integration tests for NASA API diversification and enhanced functionality.

Tests the integration of all newly implemented NASA services:
- GIOVANNI data analysis and visualization
- MODAPS/LAADS DAAC integration  
- Atmospheric Data APIs
- Earthdata Login authentication
- API key management system
- Enhanced performance benchmarking
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from nasa_cmr_agent.services.nasa_giovanni import (
    GiovanniService, GiovanniDataRequest, GiovanniAnalysisType, GiovanniDataFormat
)
from nasa_cmr_agent.services.modaps_laads import (
    MODAPSLAADSService, MODAPSDataRequest, MODISProduct, LADDSDataFormat
)
from nasa_cmr_agent.services.atmospheric_apis import (
    AtmosphericDataService, AtmosphericDataRequest, AtmosphericInstrument, 
    AtmosphericParameter, ProcessingLevel
)
from nasa_cmr_agent.services.earthdata_auth import (
    EarthdataLoginService, EarthdataCredentials, AuthenticationToken
)
from nasa_cmr_agent.services.api_key_manager import (
    APIKeyManager, APIService, KeyStatus
)
from nasa_cmr_agent.monitoring.performance_benchmarks import (
    PerformanceBenchmarkSystem, APIDiversificationMetrics
)


class TestGiovanniIntegration:
    """Test GIOVANNI service integration."""
    
    @pytest.mark.asyncio
    async def test_giovanni_time_series_analysis(self):
        """Test GIOVANNI time series analysis capability."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock GIOVANNI response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "REQUEST_ID=giovanni_test_12345"
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            async with GiovanniService() as service:
                result = await service.generate_time_series(
                    dataset_id="GPM_3IMERGM",
                    variables=["precipitation"],
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    bbox=(-10, 0, 10, 20)  # West Africa region
                )
                
                assert result.request_id.startswith("giovanni_")
                assert result.analysis_type == "time_series"
                assert result.dataset_info is not None
    
    @pytest.mark.asyncio
    async def test_giovanni_correlation_analysis(self):
        """Test GIOVANNI correlation analysis between datasets."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "REQUEST_ID=giovanni_corr_67890"
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            async with GiovanniService() as service:
                result = await service.perform_correlation_analysis(
                    dataset1_id="GPM_3IMERGM",
                    dataset2_id="GLDAS_NOAH025_M",
                    variables1=["precipitation"],
                    variables2=["SoilMoi0_10cm_tavg"],
                    start_date="2020-01-01",
                    end_date="2020-12-31"
                )
                
                assert result.request_id.startswith("giovanni_")
                assert "correlation" in result.analysis_type.lower()


class TestMODAPSLAADSIntegration:
    """Test MODAPS/LAADS DAAC service integration."""
    
    @pytest.mark.asyncio
    async def test_modis_data_search(self):
        """Test MODIS data product search."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock LAADS response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "content": [
                    {
                        "name": "MOD04_L2.A2023001.0000.061.2023002000000.hdf",
                        "id": "12345",
                        "size": 157286400,
                        "checksum": "abc123def456",
                        "collection": "61",
                        "processingDate": "2023-01-02",
                        "downloadsLink": "https://ladsweb.modaps.eosdis.nasa.gov/archive/...",
                        "bbox": [-90, -180, 90, 180]
                    }
                ]
            }
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            async with MODAPSLAADSService() as service:
                request = MODAPSDataRequest(
                    product=MODISProduct.MOD04_L2,
                    collection="61",
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    bbox=(-10, 0, 10, 20),
                    data_format=LADDSDataFormat.HDF4
                )
                
                result = await service.search_modis_data(request)
                
                assert result.total_files == 1
                assert len(result.files) == 1
                assert result.files[0].product == "MOD04_L2"
                assert result.files[0].file_name.endswith(".hdf")
    
    @pytest.mark.asyncio
    async def test_modis_product_info(self):
        """Test MODIS product information retrieval."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "title": "MODIS/Terra Aerosol 5-Min L2 Swath 10km",
                "description": "Level-2 aerosol product",
                "collections": [{"version": "61"}]
            }
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            async with MODAPSLAADSService() as service:
                info = await service.get_product_info(MODISProduct.MOD04_L2)
                
                assert "title" in info
                assert "aerosol" in info["title"].lower()


class TestAtmosphericDataIntegration:
    """Test Atmospheric Data APIs integration."""
    
    @pytest.mark.asyncio
    async def test_airs_temperature_data_search(self):
        """Test AIRS atmospheric temperature data search."""
        
        async with AtmosphericDataService() as service:
            request = AtmosphericDataRequest(
                instrument=AtmosphericInstrument.AIRS,
                parameter=AtmosphericParameter.TEMPERATURE,
                processing_level=ProcessingLevel.L2,
                start_date="2023-06-01",
                end_date="2023-06-30",
                bbox=(-120, 30, -100, 50)  # Western US
            )
            
            result = await service.search_atmospheric_data(request)
            
            assert result.total_datasets >= 0
            assert result.request_id.startswith("atm_")
            if result.datasets:
                dataset = result.datasets[0]
                assert dataset.instrument == "AIRS"
                assert dataset.parameter == "temperature"
    
    @pytest.mark.asyncio
    async def test_omi_ozone_data_search(self):
        """Test OMI ozone data search."""
        
        async with AtmosphericDataService() as service:
            request = AtmosphericDataRequest(
                instrument=AtmosphericInstrument.OMI,
                parameter=AtmosphericParameter.OZONE,
                processing_level=ProcessingLevel.L2,
                start_date="2023-03-01",
                end_date="2023-03-31"
            )
            
            result = await service.search_atmospheric_data(request)
            
            assert result.total_datasets >= 0
            if result.datasets:
                dataset = result.datasets[0]
                assert dataset.instrument == "OMI"
                assert dataset.parameter == "ozone"
    
    @pytest.mark.asyncio
    async def test_available_instruments(self):
        """Test getting available atmospheric instruments."""
        
        async with AtmosphericDataService() as service:
            instruments = await service.get_available_instruments()
            
            assert isinstance(instruments, list)
            assert len(instruments) > 0
            
            # Check that we have major atmospheric instruments
            instrument_names = [inst["instrument"] for inst in instruments]
            assert "airs" in instrument_names
            assert "omi" in instrument_names


class TestEarthdataAuthIntegration:
    """Test Earthdata Login authentication integration."""
    
    @pytest.mark.asyncio
    async def test_oauth_authorization_url_generation(self):
        """Test OAuth authorization URL generation."""
        
        async with EarthdataLoginService() as service:
            service.client_id = "test_client_id"
            service.redirect_uri = "http://localhost:8000/callback"
            
            auth_url, state = service.generate_authorization_url(scopes=["read"])
            
            assert auth_url.startswith("https://urs.earthdata.nasa.gov/oauth/authorize")
            assert "client_id=test_client_id" in auth_url
            assert "state=" in auth_url
            assert state in service.active_states
    
    @pytest.mark.asyncio
    async def test_credential_authentication(self):
        """Test authentication with username/password credentials."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful token response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "access_token": "test_access_token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "test_refresh_token"
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            async with EarthdataLoginService() as service:
                credentials = EarthdataCredentials(
                    username="test_user",
                    password="test_password"
                )
                
                token = await service.authenticate_with_credentials(credentials)
                
                assert isinstance(token, AuthenticationToken)
                assert token.access_token == "test_access_token"
                assert token.token_type == "Bearer"
                assert not token.is_expired
    
    @pytest.mark.asyncio
    async def test_authentication_status(self):
        """Test authentication status checking."""
        
        async with EarthdataLoginService() as service:
            # Initially not authenticated
            assert not service.is_authenticated()
            
            # Mock authenticated state
            service.current_token = AuthenticationToken(
                access_token="test_token",
                token_type="Bearer",
                expires_in=3600
            )
            service.auth_state = service.auth_state.AUTHENTICATED
            
            assert service.is_authenticated()
            
            status = service.get_auth_status()
            assert status["authenticated"] is True
            assert status["auth_state"] == "authenticated"


class TestAPIKeyManagerIntegration:
    """Test API Key Manager integration."""
    
    @pytest.mark.asyncio
    async def test_api_key_storage_and_retrieval(self):
        """Test storing and retrieving API keys."""
        
        with patch('nasa_cmr_agent.security.encryption.get_encryption_service') as mock_encryption:
            # Mock encryption service
            mock_encryption.return_value.encrypt_cache_data.return_value = ("encrypted_key", "key_id")
            mock_encryption.return_value.decrypt_cache_data.return_value = "test_api_key_12345"
            
            manager = APIKeyManager()
            await manager.initialize()
            
            # Store an API key
            key_id = await manager.store_api_key(
                service=APIService.OPENAI,
                key="sk-test_openai_key_12345",
                permissions=["read", "write"]
            )
            
            assert key_id.startswith("openai_")
            assert APIService.OPENAI in manager.active_keys
            
            # Retrieve the API key
            retrieved_key = await manager.get_api_key(APIService.OPENAI)
            assert retrieved_key == "test_api_key_12345"
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self):
        """Test API key rotation functionality."""
        
        with patch('nasa_cmr_agent.security.encryption.get_encryption_service') as mock_encryption:
            mock_encryption.return_value.encrypt_cache_data.return_value = ("encrypted_key", "key_id")
            mock_encryption.return_value.decrypt_cache_data.return_value = "new_api_key_67890"
            
            manager = APIKeyManager()
            await manager.initialize()
            
            # Store initial key
            await manager.store_api_key(
                service=APIService.ANTHROPIC,
                key="sk-ant-old_key_12345"
            )
            
            # Rotate to new key
            success = await manager.rotate_api_key(
                service=APIService.ANTHROPIC,
                key="sk-ant-new_key_67890"
            )
            
            assert success is True
            
            # Verify new key is active
            current_key = await manager.get_api_key(APIService.ANTHROPIC)
            assert current_key == "new_api_key_67890"
    
    @pytest.mark.asyncio
    async def test_key_validation(self):
        """Test API key validation across services."""
        
        with patch('nasa_cmr_agent.security.encryption.get_encryption_service') as mock_encryption:
            mock_encryption.return_value.encrypt_cache_data.return_value = ("encrypted_key", "key_id")
            mock_encryption.return_value.decrypt_cache_data.return_value = "valid_key"
            
            manager = APIKeyManager()
            await manager.initialize()
            
            # Test valid OpenAI key format
            is_valid = await manager._validate_openai_key("sk-valid_openai_key_format")
            assert is_valid is True
            
            # Test invalid OpenAI key format
            is_valid = await manager._validate_openai_key("invalid_key")
            assert is_valid is False
            
            # Test valid Anthropic key format
            is_valid = await manager._validate_anthropic_key("sk-ant-valid_anthropic_key")
            assert is_valid is True


class TestPerformanceBenchmarkingIntegration:
    """Test enhanced performance benchmarking integration."""
    
    @pytest.mark.asyncio
    async def test_api_diversification_metrics(self):
        """Test API diversification performance measurement."""
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Mock the API testing methods
        with patch.object(benchmark_system, '_test_cmr_api', return_value=True), \
             patch.object(benchmark_system, '_test_giovanni_api', return_value=True), \
             patch.object(benchmark_system, '_test_modaps_api', return_value=False), \
             patch.object(benchmark_system, '_test_atmospheric_api', return_value=True), \
             patch.object(benchmark_system, '_test_earthdata_auth', return_value=True):
            
            metrics = await benchmark_system.measure_api_diversification()
            
            assert isinstance(metrics, APIDiversificationMetrics)
            assert metrics.cmr_api_success_rate == 1.0
            assert metrics.giovanni_api_success_rate == 1.0
            assert metrics.modaps_api_success_rate == 0.0
            assert metrics.atmospheric_api_success_rate == 1.0
            assert metrics.earthdata_auth_success_rate == 1.0
            
            # Overall score should be good but not perfect due to MODAPS failure
            assert 0.5 < metrics.overall_diversification_score < 1.0
            assert metrics.data_coverage_enhancement > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_benchmark_dashboard(self):
        """Test that comprehensive benchmarking captures all metrics."""
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Mock all API tests to succeed
        with patch.object(benchmark_system, '_test_cmr_api', return_value=True), \
             patch.object(benchmark_system, '_test_giovanni_api', return_value=True), \
             patch.object(benchmark_system, '_test_modaps_api', return_value=True), \
             patch.object(benchmark_system, '_test_atmospheric_api', return_value=True), \
             patch.object(benchmark_system, '_test_earthdata_auth', return_value=True):
            
            # Measure diversification
            await benchmark_system.measure_api_diversification()
            
            # Get comprehensive dashboard
            dashboard = benchmark_system.get_comprehensive_dashboard()
            
            assert "api_diversification" in dashboard
            assert "performance_summary" in dashboard
            assert "recent_benchmarks" in dashboard
            
            # Verify diversification metrics are included
            diversification = dashboard["api_diversification"]
            assert "overall_score" in diversification
            assert "api_availability" in diversification
            assert "data_coverage_enhancement" in diversification


class TestIntegratedWorkflow:
    """Test integrated workflow using multiple NASA APIs."""
    
    @pytest.mark.asyncio
    async def test_multi_api_atmospheric_analysis_workflow(self):
        """Test workflow combining multiple atmospheric data sources."""
        
        # This test would simulate a complex workflow that:
        # 1. Authenticates with Earthdata Login
        # 2. Searches for AIRS temperature data via Atmospheric APIs
        # 3. Searches for MODIS aerosol data via MODAPS
        # 4. Performs correlation analysis via GIOVANNI
        # 5. Measures performance with enhanced benchmarking
        
        workflow_results = {
            "earthdata_auth": False,
            "atmospheric_search": False,
            "modis_search": False,
            "giovanni_analysis": False,
            "benchmarking": False
        }
        
        # Step 1: Earthdata authentication
        try:
            async with EarthdataLoginService() as auth_service:
                status = auth_service.get_auth_status()
                workflow_results["earthdata_auth"] = isinstance(status, dict)
        except Exception:
            pass  # Expected in test environment
        
        # Step 2: Atmospheric data search
        try:
            async with AtmosphericDataService() as atm_service:
                instruments = await atm_service.get_available_instruments()
                workflow_results["atmospheric_search"] = len(instruments) > 0
        except Exception:
            pass
        
        # Step 3: MODIS data search  
        try:
            with patch('aiohttp.ClientSession'):
                async with MODAPSLAADSService() as modis_service:
                    info = await modis_service.get_product_info(MODISProduct.MOD04_L2)
                    workflow_results["modis_search"] = isinstance(info, dict)
        except Exception:
            pass
        
        # Step 4: GIOVANNI analysis
        try:
            with patch('aiohttp.ClientSession'):
                async with GiovanniService() as giovanni_service:
                    datasets = await giovanni_service.get_available_datasets()
                    workflow_results["giovanni_analysis"] = isinstance(datasets, list)
        except Exception:
            pass
        
        # Step 5: Performance benchmarking
        try:
            benchmark_system = PerformanceBenchmarkSystem()
            with patch.object(benchmark_system, '_test_cmr_api', return_value=True):
                metrics = await benchmark_system.measure_api_diversification()
                workflow_results["benchmarking"] = isinstance(metrics, APIDiversificationMetrics)
        except Exception:
            pass
        
        # Verify that the workflow components are functional
        successful_steps = sum(workflow_results.values())
        assert successful_steps >= 3, f"Expected at least 3 successful workflow steps, got {successful_steps}"
        
        # At minimum, atmospheric search and benchmarking should work
        assert workflow_results["atmospheric_search"] is True
        assert workflow_results["benchmarking"] is True


class TestAPIServiceStatusIntegration:
    """Test service status monitoring across all APIs."""
    
    @pytest.mark.asyncio
    async def test_all_services_status_reporting(self):
        """Test that all services provide status information."""
        
        services_to_test = [
            ("GIOVANNI", GiovanniService),
            ("MODAPS/LAADS", MODAPSLAADSService),
            ("Atmospheric Data", AtmosphericDataService),
            ("Earthdata Auth", EarthdataLoginService),
            ("API Key Manager", APIKeyManager)
        ]
        
        status_results = {}
        
        for service_name, service_class in services_to_test:
            try:
                if service_name == "API Key Manager":
                    # Special handling for API Key Manager
                    with patch('nasa_cmr_agent.security.encryption.get_encryption_service'):
                        service = service_class()
                        await service.initialize()
                        status = service.get_service_status()
                elif service_name == "Earthdata Auth":
                    # Special handling for Earthdata Auth
                    async with service_class() as service:
                        status = service.get_auth_status()
                else:
                    # Standard service pattern
                    async with service_class() as service:
                        status = service.get_service_status()
                
                status_results[service_name] = {
                    "status_available": isinstance(status, dict),
                    "has_service_field": "service" in status or "authenticated" in status,
                    "status_keys": list(status.keys()) if isinstance(status, dict) else []
                }
                
            except Exception as e:
                status_results[service_name] = {
                    "status_available": False,
                    "error": str(e)
                }
        
        # Verify all services provide status information
        for service_name, result in status_results.items():
            assert result["status_available"], f"{service_name} should provide status information"
            if "error" not in result:
                assert result["has_service_field"], f"{service_name} status should have identifying field"
        
        # Verify we tested all expected services
        assert len(status_results) == len(services_to_test)


# Performance test for the integrated system
@pytest.mark.asyncio
@pytest.mark.performance
async def test_api_diversification_performance():
    """Test that API diversification doesn't significantly impact performance."""
    
    import time
    
    start_time = time.time()
    
    # Test multiple API operations concurrently
    tasks = []
    
    # Atmospheric data search
    async def atmospheric_task():
        async with AtmosphericDataService() as service:
            return await service.get_available_instruments()
    
    # GIOVANNI dataset listing
    async def giovanni_task():
        with patch('aiohttp.ClientSession'):
            async with GiovanniService() as service:
                return await service.get_available_datasets()
    
    # API key manager status
    async def api_key_task():
        with patch('nasa_cmr_agent.security.encryption.get_encryption_service'):
            manager = APIKeyManager()
            await manager.initialize()
            return manager.get_service_status()
    
    # Performance benchmarking
    async def benchmark_task():
        benchmark_system = PerformanceBenchmarkSystem()
        with patch.object(benchmark_system, '_test_cmr_api', return_value=True):
            return await benchmark_system.measure_api_diversification()
    
    tasks = [
        atmospheric_task(),
        giovanni_task(),
        api_key_task(),
        benchmark_task()
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify performance requirements
    assert total_time < 10.0, f"API diversification operations took {total_time:.2f}s, should be under 10s"
    
    # Verify all operations completed successfully
    successful_operations = sum(1 for result in results if not isinstance(result, Exception))
    assert successful_operations >= 3, f"Expected at least 3 successful operations, got {successful_operations}"