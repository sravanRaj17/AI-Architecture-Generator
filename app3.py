"""
AI Architecture Generator - From Idea to Deployable AI Solution
Streamlit Application: app.py

A fully functional architecture system that converts natural language
AI ideas into structured, deployable architecture blueprints.

Author: Architecture Systems Team
Company: IBM Hackathon - "AI Demystified — From Idea to Deployment"
"""

# ============================================================================
# 1. IMPORTS & CONFIGURATION
# ============================================================================

import streamlit as st
import json
import time
import datetime
from typing import Dict, List, Optional, Any  # <-- List is imported here
import pandas as pd
from dataclasses import dataclass, asdict
import random
import uuid

# Streamlit page configuration
st.set_page_config(
    page_title="AI Architecture Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global configuration
DEMO_MODE = True  # Set to False when integrating with real IBM watsonx APIs
VERSION = "1.0.0"

# ============================================================================
# 2. DATA CLASSES FOR STRUCTURED OUTPUTS
# ============================================================================

@dataclass
class IdeaAnalysis:
    """Structured output from Idea Analysis Agent"""
    domain: str
    problem_type: str
    ai_feasibility: str
    complexity_score: int
    key_requirements: List[str]
    constraints: List[str]
    
@dataclass
class DataStrategy:
    """Structured output from Data Strategy Agent"""
    data_type: str
    required_features: List[str]
    estimated_data_size: str
    data_sources: List[str]
    preprocessing_steps: List[str]
    quality_requirements: Dict[str, str]
    
@dataclass
class ModelDecision:
    """Structured output from Model Decision Agent"""
    selected_approach: str
    rejected_approaches: List[Dict[str, str]]
    justification: str
    model_framework: str
    training_requirements: Dict[str, Any]
    cost_estimate: str
    
@dataclass
class DeploymentPlan:
    """Structured output from Deployment Planning Agent"""
    deployment_style: str
    cloud_components: List[Dict[str, str]]
    runtime_requirements: Dict[str, str]
    scaling_strategy: str
    monitoring_plan: List[str]
    api_specification: Optional[Dict[str, Any]]
    
@dataclass
class Documentation:
    """Structured output from Documentation & Cost Agent"""
    architecture_summary: str
    cost_tier: str
    estimated_monthly_cost: str
    risks: List[Dict[str, str]]
    limitations: List[str]
    recommendations: List[str]
    implementation_timeline: Dict[str, str]

@dataclass
class AgentExecution:
    """Track agent execution status and results"""
    agent_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[float]
    end_time: Optional[float]
    output: Optional[Any]
    errors: List[str]

# ============================================================================
# 3. UTILITY FUNCTIONS
# ============================================================================

def generate_id() -> str:
    """Generate unique ID for each analysis session"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid_short = uuid.uuid4().hex[:6]
    return f"ARCH_{timestamp}_{uid_short}"

def simulate_processing(duration: float = 0.5):
    """Simulate processing time for realistic agent behavior"""
    time.sleep(duration)

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        milliseconds = seconds * 1000
        return f"{milliseconds:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"

def safe_dict_get(data_dict: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default"""
    if not isinstance(data_dict, dict):
        return default
    return data_dict.get(key, default)

# ============================================================================
# 4. AGENT DEFINITIONS
# ============================================================================

def idea_analysis_agent(user_idea: str, session_id: str) -> Dict:
    """
    Agent 1: Idea Analysis Agent
    Analyzes the user's AI idea to extract domain, problem type, and feasibility
    """
    start_time = time.time()
    result = {}
    
    try:
        if DEMO_MODE:
            simulate_processing(random.uniform(0.3, 0.8))
            
            idea_lower = user_idea.lower()
            
            domain = "Technology"
            problem_type = "Classification"
            ai_feasibility = "Moderately Feasible"
            complexity_score = 5
            key_requirements = ["Accuracy", "Scalability", "Maintainability"]
            constraints = ["Budget constraints", "Time limitations"]
            
            if "health" in idea_lower or "medical" in idea_lower:
                domain = "Healthcare"
                ai_feasibility = "Highly Feasible"
                key_requirements = ["High accuracy", "HIPAA compliance", "Real-time inference"]
                constraints = ["Limited training data", "Regulatory constraints"]
            elif "image" in idea_lower or "vision" in idea_lower:
                domain = "Computer Vision"
                problem_type = "Object Detection/Classification"
                complexity_score = 7
                key_requirements = ["High precision", "Real-time processing", "GPU acceleration"]
            elif "text" in idea_lower or "document" in idea_lower or "language" in idea_lower:
                domain = "Natural Language Processing"
                problem_type = "Text Classification/Generation"
                ai_feasibility = "Feasible"
                key_requirements = ["Context understanding", "Multi-language support"]
            elif "predict" in idea_lower or "forecast" in idea_lower or "time" in idea_lower:
                domain = "Business Analytics"
                problem_type = "Regression/Time Series"
                complexity_score = 6
                key_requirements = ["High accuracy", "Interpretability", "Real-time updates"]
            
            result = {
                "domain": domain,
                "problem_type": problem_type,
                "ai_feasibility": ai_feasibility,
                "complexity_score": complexity_score,
                "key_requirements": key_requirements,
                "constraints": constraints
            }
        
        return result
        
    except Exception as e:
        return {
            "domain": "General",
            "problem_type": "Analysis",
            "ai_feasibility": "Unknown",
            "complexity_score": 5,
            "key_requirements": ["Basic functionality"],
            "constraints": ["Implementation challenges"]
        }

def data_strategy_agent(idea_analysis: Dict, session_id: str) -> Dict:
    """
    Agent 2: Data Strategy Agent
    Determines data requirements, features, and preprocessing strategy
    """
    start_time = time.time()
    result = {}
    
    try:
        if DEMO_MODE:
            simulate_processing(random.uniform(0.4, 0.9))
            
            domain = safe_dict_get(idea_analysis, "domain", "General")
            problem_type = safe_dict_get(idea_analysis, "problem_type", "Analysis")
            complexity = safe_dict_get(idea_analysis, "complexity_score", 5)
            
            data_type = "Structured Tabular"
            required_features = ["feature_1", "feature_2", "feature_3", "target_variable"]
            estimated_data_size = "10-50 GB"
            data_sources = ["Internal databases", "Public datasets", "APIs"]
            preprocessing_steps = ["Data cleaning", "Normalization", "Feature engineering", "Validation split"]
            quality_requirements = {"accuracy": ">90%", "completeness": ">85%"}
            
            if domain == "Healthcare":
                data_type = "Structured Tabular"
                required_features = ["patient_age", "symptoms", "lab_results", "medical_history", "diagnosis"]
                estimated_data_size = "10-100 GB"
                data_sources = ["Electronic Health Records", "Lab systems", "Medical imaging archives"]
                quality_requirements = {"accuracy": ">95%", "completeness": ">90%", "privacy": "HIPAA compliant"}
            elif domain == "Computer Vision":
                data_type = "Image/Video"
                required_features = ["image_pixels", "labels", "metadata"]
                estimated_data_size = "100-500 GB"
                data_sources = ["Image databases", "Video streams", "Public datasets"]
                preprocessing_steps = ["Image resizing", "Normalization", "Augmentation", "Annotation"]
            elif domain == "Natural Language Processing":
                data_type = "Text"
                required_features = ["text_content", "labels", "metadata"]
                estimated_data_size = "5-50 GB"
                data_sources = ["Document repositories", "Web scraping", "Public corpora"]
                preprocessing_steps = ["Tokenization", "Stopword removal", "Stemming", "Vectorization"]
            
            if complexity > 7:
                estimated_data_size = "500 GB - 2 TB"
                preprocessing_steps.append("Advanced feature extraction")
                preprocessing_steps.append("Dimensionality reduction")
            
            result = {
                "data_type": data_type,
                "required_features": required_features,
                "estimated_data_size": estimated_data_size,
                "data_sources": data_sources,
                "preprocessing_steps": preprocessing_steps,
                "quality_requirements": quality_requirements
            }
        
        return result
        
    except Exception as e:
        return {
            "data_type": "Structured",
            "required_features": ["basic_features"],
            "estimated_data_size": "Unknown",
            "data_sources": ["Standard sources"],
            "preprocessing_steps": ["Basic cleaning"],
            "quality_requirements": {"quality": "standard"}
        }

def model_decision_agent(idea_analysis: Dict, data_strategy: Dict, session_id: str) -> Dict:
    """
    Agent 3: Model Decision Agent
    Selects optimal ML approach with explicit justification for rejected alternatives
    """
    start_time = time.time()
    result = {}
    
    try:
        if DEMO_MODE:
            simulate_processing(random.uniform(0.5, 1.0))
            
            domain = safe_dict_get(idea_analysis, "domain", "General")
            problem_type = safe_dict_get(idea_analysis, "problem_type", "Classification")
            data_type = safe_dict_get(data_strategy, "data_type", "Structured Tabular")
            complexity = safe_dict_get(idea_analysis, "complexity_score", 5)
            
            selected_approach = "Machine Learning"
            rejected_approaches = []
            justification = ""
            model_framework = "Scikit-learn"
            training_requirements = {}
            cost_estimate = ""
            
            data_type_lower = data_type.lower()
            problem_type_lower = problem_type.lower()
            
            if "image" in data_type_lower or "video" in data_type_lower:
                selected_approach = "Deep Learning (CNN)"
                rejected_approaches = [
                    {"approach": "Traditional ML", "reason": "Ineffective for image pattern recognition"},
                    {"approach": "LLM", "reason": "Not optimized for visual data"}
                ]
                justification = "Convolutional Neural Networks provide state-of-the-art performance for image analysis tasks"
                model_framework = "TensorFlow/PyTorch"
                training_requirements = {
                    "compute": "High (GPU required)",
                    "training_time": "8-24 hours",
                    "hyperparameter_tuning": "Extensive tuning needed"
                }
                cost_estimate = "$1000-3000/month for training and inference"
            elif "text" in data_type_lower or "nlp" in domain.lower():
                selected_approach = "Transformer-based Models"
                rejected_approaches = [
                    {"approach": "Traditional NLP", "reason": "Limited context understanding"},
                    {"approach": "Simple RNN", "reason": "Poor long-range dependencies"}
                ]
                justification = "Transformer architectures excel at understanding context and relationships in text data"
                model_framework = "Hugging Face Transformers"
                training_requirements = {
                    "compute": "Medium-High (GPU recommended)",
                    "training_time": "4-12 hours",
                    "hyperparameter_tuning": "Moderate tuning required"
                }
                cost_estimate = "$800-2000/month for training and inference"
            elif "classification" in problem_type_lower and "tabular" in data_type_lower:
                selected_approach = "Machine Learning"
                rejected_approaches = [
                    {"approach": "Deep Learning", "reason": "Overkill for structured tabular data"},
                    {"approach": "LLM", "reason": "Excessive cost for simple classification"}
                ]
                justification = "Traditional ML provides optimal balance of accuracy, interpretability, and cost for structured data"
                model_framework = "Scikit-learn / XGBoost"
                training_requirements = {
                    "compute": "Medium (8GB RAM, 4 cores)",
                    "training_time": "2-4 hours",
                    "hyperparameter_tuning": "Required"
                }
                cost_estimate = "$500-1000/month for training and inference"
            elif "regression" in problem_type_lower or "forecast" in problem_type_lower:
                selected_approach = "Machine Learning"
                rejected_approaches = [
                    {"approach": "Deep Learning", "reason": "Increased complexity without proportional accuracy gain"}
                ]
                justification = "Ensemble methods and regression algorithms provide reliable forecasting for business applications"
                model_framework = "Scikit-learn / LightGBM"
                training_requirements = {
                    "compute": "Medium (8GB RAM, 4 cores)",
                    "training_time": "3-6 hours",
                    "hyperparameter_tuning": "Required"
                }
                cost_estimate = "$600-1200/month for training and inference"
            
            if complexity > 8:
                cost_estimate = "$2000-5000/month for training and inference"
                training_requirements["compute"] = "High (Multiple GPUs)"
                training_requirements["training_time"] = "24-48 hours"
            
            if not justification:
                justification = f"Selected approach optimized for {domain} domain with {data_type} data"
            
            result = {
                "selected_approach": selected_approach,
                "rejected_approaches": rejected_approaches,
                "justification": justification,
                "model_framework": model_framework,
                "training_requirements": training_requirements,
                "cost_estimate": cost_estimate
            }
        
        return result
        
    except Exception as e:
        return {
            "selected_approach": "Machine Learning",
            "rejected_approaches": [{"approach": "Alternative", "reason": "Not optimal"}],
            "justification": "Default approach selected due to analysis constraints",
            "model_framework": "Standard framework",
            "training_requirements": {"compute": "Basic", "training_time": "Unknown"},
            "cost_estimate": "$500-2000/month"
        }

def deployment_planning_agent(
    idea_analysis: Dict, 
    data_strategy: Dict, 
    model_decision: Dict,
    session_id: str
) -> Dict:
    """
    Agent 4: Deployment Planning Agent
    Designs deployment architecture and cloud components
    """
    start_time = time.time()
    result = {}
    
    try:
        if DEMO_MODE:
            simulate_processing(random.uniform(0.4, 0.9))
            
            domain = safe_dict_get(idea_analysis, "domain", "General")
            complexity = safe_dict_get(idea_analysis, "complexity_score", 5)
            selected_approach = safe_dict_get(model_decision, "selected_approach", "Machine Learning")
            
            deployment_style = "API-based microservice"
            cloud_components = []
            runtime_requirements = {}
            scaling_strategy = "Horizontal auto-scaling based on request load"
            monitoring_plan = []
            api_specification = {}
            
            cloud_components = [
                {"component": "IBM Cloud Code Engine", "purpose": "Containerized model serving"},
                {"component": "IBM Cloud Object Storage", "purpose": "Model artifacts and data storage"},
                {"component": "IBM Cloud Databases", "purpose": "Feature store and metadata management"}
            ]
            
            runtime_requirements = {
                "memory": "4GB",
                "cpu": "2 vCPU",
                "storage": "20GB",
                "network": "Low latency (<100ms)"
            }
            
            monitoring_plan = [
                "Model performance drift detection",
                "API latency monitoring",
                "Error rate tracking",
                "Cost optimization alerts"
            ]
            
            api_specification = {
                "endpoint": "/api/v1/predict",
                "method": "POST",
                "input_format": "JSON",
                "response_time": "<200ms",
                "authentication": "API Key required"
            }
            
            if "Deep Learning" in selected_approach or "Transformer" in selected_approach:
                deployment_style = "GPU-accelerated microservice"
                runtime_requirements["gpu"] = "1-2 NVIDIA T4/V100"
                runtime_requirements["memory"] = "16GB"
                cloud_components.append({"component": "IBM Cloud GPU Instances", "purpose": "Model training and inference"})
                scaling_strategy = "GPU-based auto-scaling with request queueing"
            
            if domain == "Healthcare":
                cloud_components.append({"component": "IBM Cloud Security Advisor", "purpose": "HIPAA compliance monitoring"})
                monitoring_plan.append("Data privacy compliance auditing")
                api_specification["encryption"] = "End-to-end TLS 1.3"
            
            if complexity > 7:
                deployment_style = "Distributed microservices architecture"
                cloud_components.append({"component": "IBM Cloud Kubernetes Service", "purpose": "Orchestration and management"})
                scaling_strategy = "Kubernetes-based auto-scaling with pod replication"
                runtime_requirements["memory"] = "8GB"
                runtime_requirements["cpu"] = "4 vCPU"
            
            result = {
                "deployment_style": deployment_style,
                "cloud_components": cloud_components,
                "runtime_requirements": runtime_requirements,
                "scaling_strategy": scaling_strategy,
                "monitoring_plan": monitoring_plan,
                "api_specification": api_specification
            }
        
        return result
        
    except Exception as e:
        return {
            "deployment_style": "Standard deployment",
            "cloud_components": [{"component": "Basic cloud services", "purpose": "General purpose"}],
            "runtime_requirements": {"requirements": "standard"},
            "scaling_strategy": "Basic scaling",
            "monitoring_plan": ["Basic monitoring"],
            "api_specification": {"spec": "basic"}
        }

def documentation_cost_agent(
    idea_analysis: Dict,
    data_strategy: Dict,
    model_decision: Dict,
    deployment_plan: Dict,
    session_id: str
) -> Dict:
    """
    Agent 5: Documentation & Cost Agent
    Generates final architecture summary, cost estimates, and risk assessment
    """
    start_time = time.time()
    result = {}
    
    try:
        if DEMO_MODE:
            simulate_processing(random.uniform(0.6, 1.2))
            
            domain = safe_dict_get(idea_analysis, "domain", "General")
            problem_type = safe_dict_get(idea_analysis, "problem_type", "Analysis")
            complexity = safe_dict_get(idea_analysis, "complexity_score", 5)
            selected_approach = safe_dict_get(model_decision, "selected_approach", "Machine Learning")
            
            architecture_summary = ""
            cost_tier = "Medium"
            estimated_monthly_cost = ""
            risks = []
            limitations = []
            recommendations = []
            implementation_timeline = {}
            
            architecture_summary = f"A scalable {selected_approach} system for {domain} {problem_type} using IBM Cloud services"
            
            if complexity < 4:
                cost_tier = "Low"
                estimated_monthly_cost = "$200-500"
            elif complexity < 7:
                cost_tier = "Medium"
                estimated_monthly_cost = "$500-2000"
            else:
                cost_tier = "High"
                estimated_monthly_cost = "$2000-5000"
            
            risks = [
                {"risk": "Data quality issues", "mitigation": "Implement robust data validation pipeline"},
                {"risk": "Model performance drift", "mitigation": "Regular retraining schedule with monitoring"},
                {"risk": "Scalability challenges", "mitigation": "Auto-scaling configuration with load testing"}
            ]
            
            if domain == "Healthcare":
                risks.append({"risk": "Regulatory compliance", "mitigation": "HIPAA-compliant architecture with audit trail"})
            
            limitations = [
                "Requires labeled training data",
                "Performance depends on data quality",
                "Initial setup complexity"
            ]
            
            if "Deep Learning" in selected_approach:
                limitations.append("High computational resource requirements")
                limitations.append("Longer training times")
            
            recommendations = [
                "Start with pilot deployment in staging environment",
                "Implement comprehensive monitoring from day one",
                "Establish data governance policies",
                "Plan for regular model retraining"
            ]
            
            if complexity > 6:
                recommendations.append("Consider phased rollout approach")
                recommendations.append("Allocate budget for performance optimization")
            
            implementation_timeline = {
                "data_preparation": "2-3 weeks",
                "model_development": "3-4 weeks",
                "testing_validation": "2 weeks",
                "deployment": "1-2 weeks",
                "monitoring_setup": "1 week"
            }
            
            if complexity > 7:
                implementation_timeline["data_preparation"] = "3-4 weeks"
                implementation_timeline["model_development"] = "4-6 weeks"
            
            result = {
                "architecture_summary": architecture_summary,
                "cost_tier": cost_tier,
                "estimated_monthly_cost": estimated_monthly_cost,
                "risks": risks,
                "limitations": limitations,
                "recommendations": recommendations,
                "implementation_timeline": implementation_timeline
            }
        
        return result
        
    except Exception as e:
        return {
            "architecture_summary": "Basic AI architecture",
            "cost_tier": "Medium",
            "estimated_monthly_cost": "$500-2000/month",
            "risks": [{"risk": "General risks", "mitigation": "Standard mitigation"}],
            "limitations": ["Standard limitations apply"],
            "recommendations": ["Follow best practices"],
            "implementation_timeline": {"timeline": "4-6 weeks"}
        }

# ============================================================================
# 5. ORCHESTRATOR LOGIC
# ============================================================================

class AgentOrchestrator:
    """
    Central orchestrator managing agent execution flow
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.execution_log: List[AgentExecution] = []  # <-- Type hint added
        self.results = {}
        
    def execute_agent(self, agent_name: str, agent_func, *args) -> Dict:
        """Execute a single agent with proper logging"""
        
        execution = AgentExecution(
            agent_name=agent_name,
            status="running",
            start_time=time.time(),
            end_time=None,
            output=None,
            errors=[]
        )
        self.execution_log.append(execution)
        
        try:
            result = agent_func(*args, self.session_id)
            
            execution.status = "completed"
            execution.end_time = time.time()
            execution.output = result
            
            self.results[agent_name] = result
            
            return result
            
        except Exception as e:
            execution.status = "failed"
            execution.end_time = time.time()
            execution.errors.append(str(e))
            
            default_result = {}
            execution.output = default_result
            self.results[agent_name] = default_result
            
            return default_result
            
    def orchestrate(self, user_idea: str) -> Dict:
        """Execute all agents in sequence with dependency management"""
        
        st.session_state["orchestrator"] = self
        
        idea_result = self.execute_agent(
            "idea_analysis",
            idea_analysis_agent,
            user_idea
        )
        
        data_result = self.execute_agent(
            "data_strategy",
            data_strategy_agent,
            idea_result
        )
        
        model_result = self.execute_agent(
            "model_decision",
            model_decision_agent,
            idea_result,
            data_result
        )
        
        deployment_result = self.execute_agent(
            "deployment_planning",
            deployment_planning_agent,
            idea_result,
            data_result,
            model_result
        )
        
        documentation_result = self.execute_agent(
            "documentation_cost",
            documentation_cost_agent,
            idea_result,
            data_result,
            model_result,
            deployment_result
        )
        
        final_architecture = {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_idea": user_idea,
            "idea_analysis": idea_result,
            "data_strategy": data_result,
            "model_decision": model_result,
            "deployment_plan": deployment_result,
            "documentation": documentation_result,
            "execution_summary": self.get_execution_summary()
        }
        
        return final_architecture
    
    def get_execution_summary(self) -> Dict:
        """Generate summary of agent execution"""
        total_agents = len(self.execution_log)
        successful_agents = 0
        failed_agents = 0
        total_duration = 0.0
        agent_details = []
        
        for exec_record in self.execution_log:
            if exec_record.status == "completed":
                successful_agents += 1
            elif exec_record.status == "failed":
                failed_agents += 1
            
            duration = 0.0
            if exec_record.start_time and exec_record.end_time:
                duration = exec_record.end_time - exec_record.start_time
                total_duration += duration
            
            agent_details.append({
                "agent": exec_record.agent_name,
                "status": exec_record.status,
                "duration": format_duration(duration)
            })
        
        return {
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "total_duration": format_duration(total_duration),
            "agent_details": agent_details
        }

# ============================================================================
# 6. STREAMLIT UI LAYOUT
# ============================================================================

def render_header():
    """Render application header"""
    header_html = """
    <div style='background: linear-gradient(90deg, #1F3A93 0%, #0062FF 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>AI Architecture Generator</h1>
        <h3 style='color: #E0E0FF; margin: 0.5rem 0;'>From Idea to Deployable AI Solution</h3>
        <p style='color: #B0B0FF; margin: 0; font-size: 1.1rem;'>
            <strong>IBM Architecture Generator</strong> | 
            IBM Hackathon: "AI Demystified — From Idea to Deployment"
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_idea_input():
    """Render user input section"""
    st.markdown("### Step 1: Describe Your AI Idea")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        idea = st.text_area(
            "Describe your AI idea in natural language:",
            placeholder="Example: 'Build an AI system that analyzes medical images to detect early signs of cancer...'",
            height=120,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button(
            "Generate AI Architecture",
            type="primary",
            use_container_width=True
        )
    
    return idea, generate_btn

def render_agent_tracker(execution_log: List[AgentExecution]):  # <-- This should now work
    """Render agent execution status tracker"""
    st.markdown("### Step 2: Architecture Analysis Progress")
    
    cols = st.columns(5)
    
    agent_display_names = {
        "idea_analysis": "Idea Analysis",
        "data_strategy": "Data Strategy",
        "model_decision": "Model Decision",
        "deployment_planning": "Deployment Planning",
        "documentation_cost": "Documentation"
    }
    
    for idx, (agent_key, display_name) in enumerate(agent_display_names.items()):
        with cols[idx]:
            exec_record = None
            for e in execution_log:
                if e.agent_name == agent_key:
                    exec_record = e
                    break
            
            if not exec_record:
                status = "Pending"
                color = "#666666"
            elif exec_record.status == "running":
                status = "Running"
                color = "#FFA500"
            elif exec_record.status == "completed":
                status = "Completed"
                color = "#00CC00"
            else:
                status = "Failed"
                color = "#FF4444"
            
            agent_html = f"""
            <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; 
                        border-left: 5px solid {color}; margin-bottom: 1rem;'>
                <h4 style='color: white; margin: 0 0 0.5rem 0;'>{display_name}</h4>
                <p style='color: {color}; margin: 0; font-weight: bold;'>{status}</p>
            </div>
            """
            st.markdown(agent_html, unsafe_allow_html=True)
            
            if exec_record and exec_record.start_time and exec_record.end_time:
                duration = exec_record.end_time - exec_record.start_time
                st.caption(f"Duration: {format_duration(duration)}")

def render_decision_log(execution_log: List[AgentExecution]):
    """Render decision log panel showing agent reasoning"""
    st.markdown("### Step 3: Technical Analysis Details")
    
    for exec_record in execution_log:
        if exec_record.status == "completed" and exec_record.output:
            with st.expander(f"{exec_record.agent_name.replace('_', ' ').title()} Analysis"):
                if exec_record.agent_name == "model_decision":
                    data = exec_record.output
                    rejected_approaches = safe_dict_get(data, "rejected_approaches", [])
                    justification = safe_dict_get(data, "justification", "")
                    selected_approach = safe_dict_get(data, "selected_approach", "")
                    
                    if rejected_approaches:
                        st.markdown("**Alternative Approaches Considered:**")
                        for rejection in rejected_approaches:
                            approach = safe_dict_get(rejection, "approach", "N/A")
                            reason = safe_dict_get(rejection, "reason", "")
                            st.markdown(f"- **{approach}**: {reason}")
                    
                    if justification and selected_approach:
                        st.markdown(f"**Selected Approach:** **{selected_approach}**")
                        st.info(justification)
                
                st.json(exec_record.output)

def render_architecture_dashboard(final_architecture: Dict):
    """Render final architecture dashboard"""
    st.markdown("### Step 4: Generated AI Architecture Blueprint")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem Analysis",
        "Data Strategy", 
        "Model Design",
        "Deployment Plan",
        "Cost & Risks"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            domain = safe_dict_get(final_architecture["idea_analysis"], "domain", "N/A")
            feasibility = safe_dict_get(final_architecture["idea_analysis"], "ai_feasibility", "N/A")
            st.metric("Domain", domain)
            st.metric("Feasibility", feasibility)
        with col2:
            problem_type = safe_dict_get(final_architecture["idea_analysis"], "problem_type", "N/A")
            complexity = safe_dict_get(final_architecture["idea_analysis"], "complexity_score", "N/A")
            st.metric("Problem Type", problem_type)
            st.metric("Complexity", complexity)
        
        st.markdown("**Key Requirements:**")
        key_reqs = safe_dict_get(final_architecture["idea_analysis"], "key_requirements", [])
        for req in key_reqs:
            st.markdown(f"- {req}")
    
    with tab2:
        data_type = safe_dict_get(final_architecture["data_strategy"], "data_type", "N/A")
        estimated_size = safe_dict_get(final_architecture["data_strategy"], "estimated_data_size", "N/A")
        st.metric("Data Type", data_type)
        st.metric("Estimated Size", estimated_size)
        
        st.markdown("**Required Features:**")
        features = safe_dict_get(final_architecture["data_strategy"], "required_features", [])
        if features:
            features_df = pd.DataFrame({"Feature": features})
            st.dataframe(features_df, use_container_width=True)
        
        st.markdown("**Processing Steps:**")
        steps = safe_dict_get(final_architecture["data_strategy"], "preprocessing_steps", [])
        for step in steps:
            st.markdown(f"- {step}")
    
    with tab3:
        selected_approach = safe_dict_get(final_architecture["model_decision"], "selected_approach", "N/A")
        st.markdown(f"## **Selected Approach: {selected_approach}**")
        
        justification = safe_dict_get(final_architecture["model_decision"], "justification", "")
        if justification:
            st.info(justification)
        
        col1, col2 = st.columns(2)
        with col1:
            framework = safe_dict_get(final_architecture["model_decision"], "model_framework", "N/A")
            st.markdown("**Framework:**")
            st.code(framework)
        with col2:
            cost_estimate = safe_dict_get(final_architecture["model_decision"], "cost_estimate", "N/A")
            st.markdown("**Cost Estimate:**")
            st.success(cost_estimate)
    
    with tab4:
        deployment_style = safe_dict_get(final_architecture["deployment_plan"], "deployment_style", "N/A")
        st.metric("Deployment Style", deployment_style)
        
        st.markdown("**Cloud Components:**")
        components = safe_dict_get(final_architecture["deployment_plan"], "cloud_components", [])
        for component in components:
            comp_name = safe_dict_get(component, "component", "")
            comp_purpose = safe_dict_get(component, "purpose", "")
            st.markdown(f"- **{comp_name}**: {comp_purpose}")
        
        scaling_strategy = safe_dict_get(final_architecture["deployment_plan"], "scaling_strategy", "N/A")
        st.markdown("**Scaling Strategy:**")
        st.info(scaling_strategy)
    
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            cost_tier = safe_dict_get(final_architecture["documentation"], "cost_tier", "N/A")
            monthly_cost = safe_dict_get(final_architecture["documentation"], "estimated_monthly_cost", "N/A")
            st.metric("Cost Tier", cost_tier)
            st.metric("Monthly Cost", monthly_cost)
        
        st.markdown("**Risks & Mitigations:**")
        risks = safe_dict_get(final_architecture["documentation"], "risks", [])
        for risk in risks:
            risk_desc = safe_dict_get(risk, "risk", "")
            mitigation = safe_dict_get(risk, "mitigation", "")
            st.markdown(f"**Risk**: {risk_desc}")
            st.markdown(f"  *Mitigation*: {mitigation}")
        
        st.markdown("**Implementation Timeline:**")
        timeline = safe_dict_get(final_architecture["documentation"], "implementation_timeline", {})
        for phase, duration in timeline.items():
            phase_display = phase.replace('_', ' ').title()
            st.markdown(f"- **{phase_display}**: {duration}")

def render_download_section(final_architecture: Dict):
    """Render download options for the architecture"""
    st.markdown("### Step 5: Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_str = json.dumps(final_architecture, indent=2)
        session_id = safe_dict_get(final_architecture, "session_id", "architecture")
        st.download_button(
            label="Download JSON Blueprint",
            data=json_str,
            file_name=f"ai_architecture_{session_id}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        md_content = generate_markdown_summary(final_architecture)
        st.download_button(
            label="Download Report",
            data=md_content,
            file_name=f"ai_architecture_report_{session_id}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        if st.button("Generate Deployment Script", use_container_width=True):
            script = generate_deployment_script(final_architecture)
            st.code(script, language="bash")

def generate_markdown_summary(architecture: Dict) -> str:
    """Generate markdown summary of the architecture"""
    session_id = safe_dict_get(architecture, "session_id", "Unknown")
    timestamp = safe_dict_get(architecture, "timestamp", "Unknown")
    user_idea = safe_dict_get(architecture, "user_idea", "")
    
    idea_analysis = safe_dict_get(architecture, "idea_analysis", {})
    data_strategy = safe_dict_get(architecture, "data_strategy", {})
    model_decision = safe_dict_get(architecture, "model_decision", {})
    deployment_plan = safe_dict_get(architecture, "deployment_plan", {})
    documentation = safe_dict_get(architecture, "documentation", {})
    
    domain = safe_dict_get(idea_analysis, "domain", "N/A")
    problem_type = safe_dict_get(idea_analysis, "problem_type", "N/A")
    feasibility = safe_dict_get(idea_analysis, "ai_feasibility", "N/A")
    complexity = safe_dict_get(idea_analysis, "complexity_score", "N/A")
    
    data_type = safe_dict_get(data_strategy, "data_type", "N/A")
    estimated_size = safe_dict_get(data_strategy, "estimated_data_size", "N/A")
    features = safe_dict_get(data_strategy, "required_features", [])
    features_str = ", ".join(features) if features else "None"
    
    selected_approach = safe_dict_get(model_decision, "selected_approach", "N/A")
    framework = safe_dict_get(model_decision, "model_framework", "N/A")
    justification = safe_dict_get(model_decision, "justification", "N/A")
    
    deployment_style = safe_dict_get(deployment_plan, "deployment_style", "N/A")
    scaling_strategy = safe_dict_get(deployment_plan, "scaling_strategy", "N/A")
    
    cost_tier = safe_dict_get(documentation, "cost_tier", "N/A")
    monthly_cost = safe_dict_get(documentation, "estimated_monthly_cost", "N/A")
    
    md = f"""# AI Architecture Blueprint

## Session Details
- **Session ID**: {session_id}
- **Generated**: {timestamp}
- **Original Idea**: {user_idea}

## 1. Problem Analysis
- **Domain**: {domain}
- **Problem Type**: {problem_type}
- **Feasibility**: {feasibility}
- **Complexity Score**: {complexity}

## 2. Data Strategy
- **Data Type**: {data_type}
- **Estimated Size**: {estimated_size}
- **Key Features**: {features_str}

## 3. Model Design
- **Selected Approach**: {selected_approach}
- **Framework**: {framework}
- **Justification**: {justification}

## 4. Deployment Plan
- **Deployment Style**: {deployment_style}
- **Scaling Strategy**: {scaling_strategy}

## 5. Cost & Risks
- **Cost Tier**: {cost_tier}
- **Estimated Monthly Cost**: {monthly_cost}

---
Generated by AI Architecture Generator
"""
    return md

def generate_deployment_script(architecture: Dict) -> str:
    """Generate sample deployment script"""
    session_id = safe_dict_get(architecture, "session_id", "unknown")
    domain = safe_dict_get(architecture["idea_analysis"], "domain", "Unknown") if isinstance(architecture.get("idea_analysis"), dict) else "Unknown"
    problem_type = safe_dict_get(architecture["idea_analysis"], "problem_type", "Solution") if isinstance(architecture.get("idea_analysis"), dict) else "Solution"
    
    script = f"""#!/bin/bash

# Deployment Script for AI Architecture: {session_id}
# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "Deploying AI Architecture: {domain} {problem_type}"

# 1. Setup IBM Cloud Environment
echo "Setting up IBM Cloud components..."
# ibmcloud login
# ibmcloud target --cf
# ibmcloud resource group-create ai-{session_id.lower()}

# 2. Deploy Storage for Data
echo "Deploying storage components..."
# ibmcloud cos create-bucket --bucket ai-data-{session_id.lower()}

# 3. Deploy Model Serving Infrastructure
echo "Deploying model serving..."
# ibmcloud code-engine project create --name ai-project-{session_id.lower()}

# 4. Configure Monitoring
echo "Configuring monitoring..."
# ibmcloud monitoring service-create ai-monitor-{session_id.lower()}

echo "Deployment script generated. Review and execute with appropriate IBM Cloud credentials."

# Note: This is a template script. Actual deployment requires:
# 1. IBM Cloud account with appropriate permissions
# 2. Customization based on specific requirements
# 3. Security and compliance review
"""
    return script

def render_footer():
    """Render application footer"""
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        disclaimer_html = f"""
        <div style='background-color: #0A1F44; padding: 1.5rem; border-radius: 10px;'>
            <p style='color: #B0B0FF; margin: 0;'>
                <strong>Disclaimer:</strong> This system analyzes ideas and generates technical architectures. 
                All outputs should be reviewed by technical experts before implementation.
            </p>
            <p style='color: #8888CC; margin: 0.5rem 0 0 0;'>
                IBM Hackathon Project | "AI Demystified — From Idea to Deployment" | v{VERSION}
            </p>
        </div>
        """
        st.markdown(disclaimer_html, unsafe_allow_html=True)
    
    with col2:
        system_html = """
        <div style='text-align: center; padding: 1rem;'>
            <p style='color: #666; margin: 0;'>Architecture Generator</p>
            <p style='color: #00CC00; font-weight: bold; margin: 0;'>Technical Analysis</p>
            <p style='color: #00CC00; font-weight: bold; margin: 0;'>Deployment Ready</p>
            <p style='color: #00CC00; font-weight: bold; margin: 0;'>IBM Cloud</p>
        </div>
        """
        st.markdown(system_html, unsafe_allow_html=True)

# ============================================================================
# 7. MAIN EXECUTION FLOW
# ============================================================================

def main():
    """Main application entry point"""
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "final_architecture" not in st.session_state:
        st.session_state.final_architecture = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    render_header()
    
    idea, generate_btn = render_idea_input()
    
    if generate_btn and idea:
        st.session_state.session_id = generate_id()
        st.session_state.processing = True
        
        orchestrator = AgentOrchestrator(st.session_state.session_id)
        
        with st.spinner("Initializing architecture analysis..."):
            try:
                final_architecture = orchestrator.orchestrate(idea)
                
                st.session_state.final_architecture = final_architecture
                st.session_state.orchestrator = orchestrator
                st.session_state.processing = False
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.processing = False
    
    elif st.session_state.processing:
        st.info("Analysis in progress... Please wait.")
        st.progress(0.5, "Analyzing requirements and generating architecture")
        return
    
    if st.session_state.final_architecture and st.session_state.orchestrator:
        render_agent_tracker(st.session_state.orchestrator.execution_log)
        
        render_decision_log(st.session_state.orchestrator.execution_log)
        
        render_architecture_dashboard(st.session_state.final_architecture)
        
        render_download_section(st.session_state.final_architecture)
        
        with st.expander("Execution Summary"):
            summary = safe_dict_get(st.session_state.final_architecture, "execution_summary", {})
            total_agents = safe_dict_get(summary, "total_agents", 0)
            successful_agents = safe_dict_get(summary, "successful_agents", 0)
            failed_agents = safe_dict_get(summary, "failed_agents", 0)
            total_duration = safe_dict_get(summary, "total_duration", "0s")
            
            st.metric("Total Agents", total_agents)
            st.metric("Successful", successful_agents)
            st.metric("Failed", failed_agents)
            st.metric("Total Duration", total_duration)
    
    render_footer()
    
    if DEMO_MODE and st.session_state.processing:
        time.sleep(0.5)
        st.rerun()

# ============================================================================
# 8. APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()