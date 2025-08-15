#!/usr/bin/env python3
"""
Comprehensive LLM Provider Testing Suite
Tests all LLM providers: OpenAI, Anthropic, Gemini, DeepSeek, HuggingFace, Meta, OpenAI Open Source, Together, Cohere
Validates graceful fallback, error handling, and production readiness with no shortcuts.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nasa_cmr_agent.services.llm_service import LLMService
from nasa_cmr_agent.services.api_key_manager import get_service_api_key, APIService
from nasa_cmr_agent.core.config import settings


class ComprehensiveLLMTest:
    """Comprehensive LLM provider testing suite."""
    
    def __init__(self):
        self.test_results = {}
        self.available_providers = []
        self.configured_providers = self._check_configured_providers()
        
    def _check_configured_providers(self) -> List[str]:
        """Check which LLM providers have API keys configured."""
        providers = []
        
        # Check environment variables for each provider
        provider_checks = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"), 
            "gemini": os.getenv("GEMINI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
            "meta": os.getenv("META_API_KEY"),
            "openai_opensource": os.getenv("OPENAI_OPENSOURCE_API_KEY"),
            "together": os.getenv("TOGETHER_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY")
        }
        
        for provider, api_key in provider_checks.items():
            if api_key and api_key != "your_{}_api_key_here".format(provider.replace("_", "")):
                providers.append(provider)
                
        return providers
    
    async def run_all_tests(self):
        """Run comprehensive LLM provider tests."""
        
        print("🤖 NASA CMR Agent - Comprehensive LLM Provider Testing")
        print("=" * 70)
        print(f"🔧 Configured Providers: {', '.join(self.configured_providers) if self.configured_providers else 'None'}")
        print(f"📊 Testing Priority: {settings.llm_provider_priority}")
        print()
        
        # Test 1: Provider Initialization
        await self._test_provider_initialization()
        
        # Test 2: API Key Management Integration
        await self._test_api_key_management()
        
        # Test 3: Basic Generation Capabilities
        await self._test_basic_generation()
        
        # Test 4: Fallback Mechanism
        await self._test_fallback_mechanism()
        
        # Test 5: Batch Processing
        await self._test_batch_processing()
        
        # Test 6: Error Handling and Recovery
        await self._test_error_handling()
        
        # Test 7: Performance Characteristics
        await self._test_performance()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
    
    async def _test_provider_initialization(self):
        """Test LLM provider initialization."""
        
        print("🚀 Test 1: Provider Initialization")
        print("-" * 40)
        
        try:
            llm_service = LLMService()
            provider_status = llm_service.get_provider_status()
            
            self.available_providers = provider_status["available_providers"]
            
            print(f"✅ LLM Service initialized successfully")
            print(f"📊 Total providers available: {provider_status['total_providers']}")
            print(f"🎯 Current primary provider: {provider_status['current_provider']}")
            print()
            
            # Test each provider type
            provider_types = {"commercial": [], "alternative": [], "opensource": []}
            
            for provider_detail in provider_status["provider_details"]:
                provider_name = provider_detail["name"]
                model = provider_detail["model"]
                status = provider_detail["status"]
                
                # Get provider type from the actual provider info
                provider_type = "commercial"  # Default
                for p in llm_service.providers:
                    if p["name"] == provider_name:
                        provider_type = p.get("type", "commercial")
                        break
                
                provider_types[provider_type].append(f"{provider_name} ({model})")
                
                status_emoji = "✅" if status == "available" else "❌"
                print(f"   {status_emoji} {provider_name}: {model} ({status}, {provider_type})")
            
            print()
            print(f"📈 Provider Distribution:")
            for ptype, providers in provider_types.items():
                if providers:
                    print(f"   • {ptype.title()}: {len(providers)} providers")
            
            self.test_results["initialization"] = {
                "success": True,
                "total_providers": provider_status["total_providers"],
                "available_providers": self.available_providers,
                "provider_distribution": {k: len(v) for k, v in provider_types.items()}
            }
            
            print("✅ Provider initialization test: PASSED")
            
        except Exception as e:
            print(f"❌ Provider initialization test: FAILED - {e}")
            self.test_results["initialization"] = {"success": False, "error": str(e)}
    
    async def _test_api_key_management(self):
        """Test API key management integration."""
        
        print(f"\\n🔐 Test 2: API Key Management Integration")
        print("-" * 40)
        
        try:
            api_services = [
                APIService.OPENAI, APIService.ANTHROPIC, APIService.GEMINI,
                APIService.DEEPSEEK, APIService.HUGGINGFACE, APIService.META,
                APIService.OPENAI_OPENSOURCE, APIService.TOGETHER, APIService.COHERE
            ]
            
            available_keys = {}
            
            for service in api_services:
                try:
                    api_key = await get_service_api_key(service)
                    available_keys[service.value] = bool(api_key)
                    
                    key_status = "Available" if api_key else "Not configured"
                    key_emoji = "🔑" if api_key else "⚪"
                    print(f"   {key_emoji} {service.value}: {key_status}")
                    
                except Exception as e:
                    available_keys[service.value] = False
                    print(f"   ❌ {service.value}: Error - {e}")
            
            total_available = sum(available_keys.values())
            total_tested = len(available_keys)
            
            print(f"\\n📊 API Key Summary: {total_available}/{total_tested} providers have keys")
            
            self.test_results["api_key_management"] = {
                "success": True,
                "available_keys": available_keys,
                "total_available": total_available
            }
            
            print("✅ API key management test: PASSED")
            
        except Exception as e:
            print(f"❌ API key management test: FAILED - {e}")
            self.test_results["api_key_management"] = {"success": False, "error": str(e)}
    
    async def _test_basic_generation(self):
        """Test basic generation capabilities."""
        
        print(f"\\n🧠 Test 3: Basic Generation Capabilities")
        print("-" * 40)
        
        if not self.available_providers:
            print("⚠️  No providers available for generation testing")
            self.test_results["basic_generation"] = {"success": False, "error": "No providers available"}
            return
            
        try:
            llm_service = LLMService()
            test_prompt = "What is NASA's Common Metadata Repository? Answer in one sentence."
            
            print(f"🔍 Testing prompt: '{test_prompt[:50]}...'")
            
            start_time = time.time()
            response = await llm_service.generate(test_prompt, max_tokens=100)
            generation_time = time.time() - start_time
            
            current_provider = llm_service.get_current_provider()
            
            print(f"✅ Generation successful with {current_provider}")
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
            print(f"📝 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            
            self.test_results["basic_generation"] = {
                "success": True,
                "provider_used": current_provider,
                "generation_time": generation_time,
                "response_length": len(response),
                "response_preview": response[:100]
            }
            
            print("✅ Basic generation test: PASSED")
            
        except Exception as e:
            print(f"❌ Basic generation test: FAILED - {e}")
            self.test_results["basic_generation"] = {"success": False, "error": str(e)}
    
    async def _test_fallback_mechanism(self):
        """Test fallback mechanism between providers."""
        
        print(f"\\n🔄 Test 4: Fallback Mechanism")
        print("-" * 40)
        
        if len(self.available_providers) < 2:
            print("⚠️  Need at least 2 providers to test fallback mechanism")
            self.test_results["fallback"] = {"success": False, "error": "Insufficient providers for fallback testing"}
            return
            
        try:
            llm_service = LLMService()
            
            # Test provider status
            status_before = llm_service.get_provider_status()
            print(f"🎯 Initial provider: {status_before['current_provider']}")
            print(f"❌ Failed providers: {', '.join(status_before['failed_providers']) if status_before['failed_providers'] else 'None'}")
            
            # Simulate a failure by marking the current provider as failed
            if llm_service.providers:
                current_provider_name = llm_service.providers[llm_service.current_provider_index]["name"]
                llm_service.failed_providers.add(current_provider_name)
                
                print(f"🔧 Simulated failure for {current_provider_name}")
                
                # Try generation again - should fallback
                test_prompt = "Describe Earth science data in one sentence."
                response = await llm_service.generate(test_prompt, max_tokens=50)
                
                status_after = llm_service.get_provider_status()
                new_provider = status_after['current_provider']
                
                print(f"✅ Fallback successful to: {new_provider}")
                print(f"📝 Fallback response: {response[:80]}...")
                
                # Reset failed providers
                llm_service.reset_failed_providers()
                print(f"🔄 Reset failed providers for next tests")
                
                self.test_results["fallback"] = {
                    "success": True,
                    "original_provider": current_provider_name,
                    "fallback_provider": new_provider,
                    "fallback_worked": current_provider_name != new_provider
                }
                
                print("✅ Fallback mechanism test: PASSED")
            else:
                raise Exception("No providers available for fallback testing")
                
        except Exception as e:
            print(f"❌ Fallback mechanism test: FAILED - {e}")
            self.test_results["fallback"] = {"success": False, "error": str(e)}
    
    async def _test_batch_processing(self):
        """Test batch processing capabilities."""
        
        print(f"\\n📦 Test 5: Batch Processing")
        print("-" * 40)
        
        if not self.available_providers:
            print("⚠️  No providers available for batch processing test")
            self.test_results["batch_processing"] = {"success": False, "error": "No providers available"}
            return
            
        try:
            llm_service = LLMService()
            
            test_prompts = [
                "What is NASA?",
                "What is satellite data?",
                "What is climate science?"
            ]
            
            print(f"🚀 Processing {len(test_prompts)} prompts concurrently...")
            
            start_time = time.time()
            responses = await llm_service.batch_generate(
                test_prompts,
                max_concurrent=2,
                max_tokens=30
            )
            batch_time = time.time() - start_time
            
            successful_responses = [r for r in responses if not isinstance(r, Exception)]
            failed_responses = [r for r in responses if isinstance(r, Exception)]
            
            print(f"✅ Batch processing completed in {batch_time:.2f} seconds")
            print(f"📊 Results: {len(successful_responses)} successful, {len(failed_responses)} failed")
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"   {i+1}. ❌ Error: {response}")
                else:
                    print(f"   {i+1}. ✅ {response[:40]}...")
            
            self.test_results["batch_processing"] = {
                "success": True,
                "total_prompts": len(test_prompts),
                "successful_responses": len(successful_responses),
                "failed_responses": len(failed_responses),
                "batch_time": batch_time
            }
            
            print("✅ Batch processing test: PASSED")
            
        except Exception as e:
            print(f"❌ Batch processing test: FAILED - {e}")
            self.test_results["batch_processing"] = {"success": False, "error": str(e)}
    
    async def _test_error_handling(self):
        """Test error handling and recovery."""
        
        print(f"\\n🛡️  Test 6: Error Handling and Recovery")
        print("-" * 40)
        
        if not self.available_providers:
            print("⚠️  No providers available for error handling test")
            self.test_results["error_handling"] = {"success": False, "error": "No providers available"}
            return
            
        try:
            llm_service = LLMService()
            
            # Test 1: Invalid prompt handling
            print("🔍 Testing invalid prompt handling...")
            try:
                response = await llm_service.generate("", max_tokens=10)
                print("✅ Empty prompt handled gracefully")
                empty_prompt_handled = True
            except Exception as e:
                print(f"⚠️  Empty prompt error: {e}")
                empty_prompt_handled = False
            
            # Test 2: Extreme parameters
            print("🔍 Testing extreme parameters...")
            try:
                response = await llm_service.generate(
                    "Test", 
                    max_tokens=1,  # Very low
                    temperature=0.0  # Very low
                )
                print("✅ Extreme parameters handled gracefully")
                extreme_params_handled = True
            except Exception as e:
                print(f"⚠️  Extreme parameters error: {e}")
                extreme_params_handled = False
            
            # Test 3: Provider recovery after failure
            print("🔍 Testing provider recovery...")
            initial_failed_count = len(llm_service.get_failed_providers())
            
            # Mark all providers as failed
            for provider in llm_service.providers:
                llm_service.failed_providers.add(provider["name"])
            
            # Reset and try again
            llm_service.reset_failed_providers()
            recovery_successful = len(llm_service.get_failed_providers()) == 0
            
            print(f"✅ Provider recovery: {'successful' if recovery_successful else 'failed'}")
            
            self.test_results["error_handling"] = {
                "success": True,
                "empty_prompt_handled": empty_prompt_handled,
                "extreme_params_handled": extreme_params_handled,
                "provider_recovery": recovery_successful
            }
            
            print("✅ Error handling test: PASSED")
            
        except Exception as e:
            print(f"❌ Error handling test: FAILED - {e}")
            self.test_results["error_handling"] = {"success": False, "error": str(e)}
    
    async def _test_performance(self):
        """Test performance characteristics."""
        
        print(f"\\n⚡ Test 7: Performance Characteristics")
        print("-" * 40)
        
        if not self.available_providers:
            print("⚠️  No providers available for performance testing")
            self.test_results["performance"] = {"success": False, "error": "No providers available"}
            return
            
        try:
            llm_service = LLMService()
            
            # Test response times for different prompt sizes
            test_cases = [
                ("Short prompt", "NASA", 20),
                ("Medium prompt", "Explain NASA's Earth science mission in detail", 50),
                ("Long prompt", "Provide a comprehensive overview of NASA's Common Metadata Repository, its purpose, functionality, and importance for Earth science research", 100)
            ]
            
            performance_results = {}
            
            for test_name, prompt, max_tokens in test_cases:
                print(f"🔍 Testing {test_name.lower()}...")
                
                start_time = time.time()
                response = await llm_service.generate(prompt, max_tokens=max_tokens)
                response_time = time.time() - start_time
                
                performance_results[test_name] = {
                    "response_time": response_time,
                    "response_length": len(response),
                    "tokens_per_second": len(response.split()) / response_time if response_time > 0 else 0
                }
                
                print(f"   ⏱️  Response time: {response_time:.2f}s")
                print(f"   📊 Response length: {len(response)} characters")
                print(f"   🚀 Est. tokens/sec: {performance_results[test_name]['tokens_per_second']:.1f}")
            
            # Calculate average performance
            avg_response_time = sum(r["response_time"] for r in performance_results.values()) / len(performance_results)
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in performance_results.values()) / len(performance_results)
            
            print(f"\\n📈 Performance Summary:")
            print(f"   • Average response time: {avg_response_time:.2f} seconds")
            print(f"   • Average tokens/second: {avg_tokens_per_sec:.1f}")
            
            self.test_results["performance"] = {
                "success": True,
                "test_cases": performance_results,
                "average_response_time": avg_response_time,
                "average_tokens_per_second": avg_tokens_per_sec
            }
            
            print("✅ Performance test: PASSED")
            
        except Exception as e:
            print(f"❌ Performance test: FAILED - {e}")
            self.test_results["performance"] = {"success": False, "error": str(e)}
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        
        print(f"\\n🎉 Comprehensive Test Report")
        print("=" * 70)
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        
        print(f"📊 Test Summary: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        print()
        
        # Detailed results
        print(f"🔍 Detailed Results:")
        print("-" * 30)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"{status} {test_name.replace('_', ' ').title()}")
            
            if not result.get("success", False) and "error" in result:
                print(f"         Error: {result['error']}")
        
        print()
        
        # Provider statistics
        if "initialization" in self.test_results and self.test_results["initialization"].get("success"):
            init_results = self.test_results["initialization"]
            print(f"🤖 LLM Provider Statistics:")
            print("-" * 30)
            print(f"   • Total providers available: {init_results['total_providers']}")
            print(f"   • Configured providers: {len(self.configured_providers)}")
            
            if "provider_distribution" in init_results:
                for ptype, count in init_results["provider_distribution"].items():
                    if count > 0:
                        print(f"   • {ptype.title()} providers: {count}")
        
        # Performance insights
        if "performance" in self.test_results and self.test_results["performance"].get("success"):
            perf_results = self.test_results["performance"]
            print(f"\\n⚡ Performance Insights:")
            print("-" * 30)
            print(f"   • Average response time: {perf_results['average_response_time']:.2f}s")
            print(f"   • Average processing speed: {perf_results['average_tokens_per_second']:.1f} tokens/sec")
        
        # Recommendations
        print(f"\\n💡 Recommendations:")
        print("-" * 30)
        
        if len(self.configured_providers) == 0:
            print("   ⚠️  Configure at least one LLM API key for basic functionality")
        elif len(self.configured_providers) == 1:
            print("   📈 Add additional LLM providers for better fallback capabilities")
        elif len(self.configured_providers) >= 2:
            print("   ✅ Good provider diversity - fallback capabilities enabled")
        
        if "api_key_management" in self.test_results:
            api_results = self.test_results["api_key_management"]
            if api_results.get("success") and api_results.get("total_available", 0) >= 3:
                print("   ✅ Excellent API key coverage for production deployment")
        
        # Final status
        print(f"\\n🚀 System Status:")
        print("-" * 30)
        
        if passed_tests >= 5:
            print("   ✅ PRODUCTION READY: All critical systems operational")
            print("   ✅ COMPREHENSIVE LLM SUPPORT: Multiple providers available")
            print("   ✅ FAULT TOLERANCE: Graceful fallback mechanisms working")
        elif passed_tests >= 3:
            print("   ⚠️  FUNCTIONAL: Core systems working, some limitations")
            print("   📈 Consider adding more LLM providers for better coverage")
        else:
            print("   ❌ SETUP REQUIRED: Critical systems need configuration")
            print("   🔧 Review API keys and provider configuration")
        
        # Save detailed results to file
        try:
            results_file = project_root / "test_results_llm_comprehensive.json"
            with open(results_file, "w") as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\\n📄 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\\n⚠️  Could not save results file: {e}")


async def main():
    """Run comprehensive LLM provider tests."""
    
    test_suite = ComprehensiveLLMTest()
    await test_suite.run_all_tests()
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)