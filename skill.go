package agentcore

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// SkillDefinition is stable metadata for one reusable skill.
type SkillDefinition struct {
	Name        string `json:"name"`
	Version     string `json:"version,omitempty"`
	Description string `json:"description,omitempty"`
}

// SkillContent is materialized when an Agent is built. Instructions are
// trusted system-prompt content; Tools and Interceptors extend the same Agent.
type SkillContent struct {
	Instructions string
	Tools        []Tool
	Interceptors []Interceptor
}

// Skill can load instructions and runtime extensions from files, a database,
// an embedded bundle, or any application-owned source.
type Skill interface {
	Definition() SkillDefinition
	Load(context.Context) (SkillContent, error)
}

// FuncSkill adapts functions or static content to Skill.
type FuncSkill struct {
	SkillDefinition SkillDefinition
	Content         SkillContent
	LoadFunc        func(context.Context) (SkillContent, error)
}

func (s FuncSkill) Definition() SkillDefinition { return s.SkillDefinition }

func (s FuncSkill) Load(ctx context.Context) (SkillContent, error) {
	if s.LoadFunc != nil {
		return s.LoadFunc(ctx)
	}
	return cloneSkillContent(s.Content), nil
}

// Builder composes an Agent from application-defined tools, skills, and
// lifecycle interceptors. A Builder is mutable and intended for startup or
// execution-scope assembly; the resulting Agent is concurrency-safe.
type Builder struct {
	config Config
	skills []Skill
}

func NewBuilder(model Model) *Builder { return &Builder{config: Config{Model: model}} }

// BuilderFromConfig starts with a defensive copy of an existing Config.
func BuilderFromConfig(config Config) *Builder {
	config.Tools = append([]Tool(nil), config.Tools...)
	config.Interceptors = append([]Interceptor(nil), config.Interceptors...)
	config.ModelOptions = cloneOptions(config.ModelOptions)
	return &Builder{config: config}
}

func (b *Builder) SystemPrompt(prompt string) *Builder {
	b.config.SystemPrompt = prompt
	return b
}

func (b *Builder) Tools(tools ...Tool) *Builder {
	b.config.Tools = append(b.config.Tools, tools...)
	return b
}

func (b *Builder) Skills(skills ...Skill) *Builder {
	b.skills = append(b.skills, skills...)
	return b
}

func (b *Builder) Use(interceptors ...Interceptor) *Builder {
	b.config.Interceptors = append(b.config.Interceptors, interceptors...)
	return b
}

// Configure exposes the complete Config for advanced policies while keeping
// normal tool/skill/interceptor composition fluent.
func (b *Builder) Configure(configure func(*Config)) *Builder {
	if configure != nil {
		configure(&b.config)
	}
	return b
}

func (b *Builder) Build(ctx context.Context) (*Agent, error) {
	if b == nil {
		return nil, errors.New("agentcore: nil Builder")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	config := b.config
	config.Tools = append([]Tool(nil), config.Tools...)
	config.Interceptors = append([]Interceptor(nil), config.Interceptors...)
	seen := make(map[string]struct{}, len(b.skills))
	for index, skill := range b.skills {
		if skill == nil {
			return nil, fmt.Errorf("agentcore: nil skill at index %d", index)
		}
		definition := skill.Definition()
		name := strings.TrimSpace(definition.Name)
		if name == "" {
			return nil, fmt.Errorf("agentcore: skill %d has no name", index)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("agentcore: duplicate skill %q", name)
		}
		seen[name] = struct{}{}
		content, err := skill.Load(ctx)
		if err != nil {
			return nil, fmt.Errorf("agentcore: load skill %q: %w", name, err)
		}
		if instructions := strings.TrimSpace(content.Instructions); instructions != "" {
			config.SystemPrompt = appendSkillInstructions(config.SystemPrompt, definition, instructions)
		}
		config.Tools = append(config.Tools, content.Tools...)
		config.Interceptors = append(config.Interceptors, content.Interceptors...)
	}
	return New(config)
}

func appendSkillInstructions(systemPrompt string, definition SkillDefinition, instructions string) string {
	heading := "<skill name=" + fmt.Sprintf("%q", definition.Name)
	if definition.Version != "" {
		heading += " version=" + fmt.Sprintf("%q", definition.Version)
	}
	heading += ">"
	block := heading + "\n" + instructions + "\n</skill>"
	if strings.TrimSpace(systemPrompt) == "" {
		return block
	}
	return strings.TrimSpace(systemPrompt) + "\n\n" + block
}

func cloneSkillContent(content SkillContent) SkillContent {
	content.Tools = append([]Tool(nil), content.Tools...)
	content.Interceptors = append([]Interceptor(nil), content.Interceptors...)
	return content
}
