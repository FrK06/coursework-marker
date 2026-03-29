"""
Agents Module - Three-agent system for KSB coursework marking.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                                │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   ANALYSIS   │───▶│   SCORING    │───▶│   FEEDBACK   │       │
│  │    AGENT     │    │    AGENT     │    │    AGENT     │       │
│  │      🔍      │    │      📊      │    │      💬      │       │
│  │              │    │              │    │              │       │
│  │ • Text       │    │ • Rubric     │    │ • Strengths  │       │
│  │ • Charts     │    │ • Weights    │    │ • Gaps       │       │
│  │ • Tables     │    │ • Grades     │    │ • Suggestions│       │
│  │ • Images     │    │ • Patterns   │    │ • Formatting │       │
│  │ • Evidence   │    │ • Brief Map  │    │ • Summary    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│                       Shared Context                             │
└─────────────────────────────────────────────────────────────────┘
"""

from .core import (
    AgentRole,
    AgentContext,
    AgentOrchestrator,
    BaseAgent,
    BaseTool,
    ToolResult
)

from .analysis_agent import (
    AnalysisAgent,
    TextAnalyzerTool,
    ChartAnalyzerTool,
    TableAnalyzerTool,
    ImageAnalyzerTool,
    SectionExtractorTool,
    EvidenceFinderTool
)

from .scoring_agent import (
    ScoringAgent,
    RubricApplierTool,
    WeightCalculatorTool,
    CriteriaCheckerTool,
    BriefMapperTool
)

from .feedback_agent import (
    FeedbackAgent,
    StrengthIdentifierTool,
    GapAnalyzerTool,
    ImprovementSuggesterTool,
    FeedbackFormatterTool,
    FEEDBACK_TEMPLATES
)


def create_agent_system(llm, embedder=None, vector_store=None, verbose: bool = False, module_code: str = "MLCC"):
    """
    Factory function to create the complete three-agent system.

    Args:
        llm: OllamaClient instance
        embedder: Optional Embedder for semantic search
        vector_store: Optional ChromaStore for evidence retrieval
        verbose: Enable verbose logging
        module_code: Module code for validation (DSP, MLCC, or AIDI)

    Returns:
        Configured AgentOrchestrator
    """
    analysis_agent = AnalysisAgent(llm, embedder, vector_store, verbose, module_code=module_code)
    scoring_agent = ScoringAgent(llm, module_code=module_code, verbose=verbose)
    feedback_agent = FeedbackAgent(llm, verbose)

    return AgentOrchestrator(
        analysis_agent=analysis_agent,
        scoring_agent=scoring_agent,
        feedback_agent=feedback_agent,
        verbose=verbose
    )


__all__ = [
    # Core
    'AgentRole',
    'AgentContext',
    'AgentOrchestrator',
    'BaseAgent',
    'BaseTool',
    'ToolResult',
    
    # Agents
    'AnalysisAgent',
    'ScoringAgent',
    'FeedbackAgent',
    
    # Analysis Tools
    'TextAnalyzerTool',
    'ChartAnalyzerTool',
    'TableAnalyzerTool',
    'ImageAnalyzerTool',
    'SectionExtractorTool',
    'EvidenceFinderTool',
    
    # Scoring Tools
    'RubricApplierTool',
    'WeightCalculatorTool',
    'CriteriaCheckerTool',
    'BriefMapperTool',
    
    # Feedback Tools
    'StrengthIdentifierTool',
    'GapAnalyzerTool',
    'ImprovementSuggesterTool',
    'FeedbackFormatterTool',
    'FEEDBACK_TEMPLATES',
    
    # Factory
    'create_agent_system'
]
