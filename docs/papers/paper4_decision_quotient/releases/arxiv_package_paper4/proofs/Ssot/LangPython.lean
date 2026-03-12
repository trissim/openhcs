/-
  Python Language Model - Formal Semantics
  
  This formalizes the subset of Python needed to PROVE (not assume):
  1. Python has definition-time hooks (__init_subclass__, metaclasses)
  2. Python has introspection (__subclasses__(), type(), dir())
  3. These execute/query at the correct times
  
  The model is a direct transcription of Python's documented semantics.
  Attack requires producing Python code that contradicts the model.
-/

import Ssot.Foundations

namespace Python

/-!
## Python Object Model (Minimal)
-/

-- Python object identifier
abbrev PyId := String

-- A Python class (simplified to relevant features)
structure PyClass where
  name : PyId
  bases : List PyId
  metaclass : Option PyId
  attributes : List (String × String)  -- name → value repr
  methods : List (String × String)     -- name → signature
  deriving DecidableEq, Repr

-- Python runtime state
structure PyRuntime where
  -- All defined classes
  classes : List PyClass
  -- Subclass registry: maps class name to list of direct subclasses
  subclass_registry : PyId → List PyId
  -- Hook call log: records when __init_subclass__ was called
  init_subclass_calls : List (PyId × PyId)  -- (parent, child)

def PyRuntime.empty : PyRuntime := {
  classes := [],
  subclass_registry := fun _ => [],
  init_subclass_calls := []
}

/-!
## Python Class Definition Semantics

From Python docs (https://docs.python.org/3/reference/datamodel.html#customizing-class-creation):
"Whenever a class inherits from another class, __init_subclass__() is called on the parent class."

This happens DURING the class statement execution, not after.
-/

-- Events that occur during class definition (in order)
inductive ClassDefEvent where
  | metacall_start : PyId → ClassDefEvent           -- Metaclass.__call__ begins
  | new_called : PyId → ClassDefEvent               -- __new__ creates class object
  | namespace_populated : PyId → ClassDefEvent      -- Class body executes
  | init_subclass_called : PyId → PyId → ClassDefEvent  -- Parent's __init_subclass__(child)
  | subclasses_updated : PyId → PyId → ClassDefEvent    -- Parent.__subclasses__() updated
  | init_called : PyId → ClassDefEvent              -- __init__ finalizes class
  | class_bound : PyId → ClassDefEvent              -- Name bound in namespace
  deriving DecidableEq, Repr

-- Execute a class statement: class Name(*bases): body
-- Returns: (updated runtime, event sequence)
def execute_class_statement (rt : PyRuntime) (name : PyId) (bases : List PyId) 
    (attrs : List (String × String)) (methods : List (String × String)) 
    : PyRuntime × List ClassDefEvent :=
  let cls : PyClass := { 
    name := name, 
    bases := bases, 
    metaclass := none,  -- Using default metaclass
    attributes := attrs,
    methods := methods
  }
  -- Event sequence (from Python semantics)
  let events := [.metacall_start name, .new_called name, .namespace_populated name]
  -- For each base: call __init_subclass__ and update __subclasses__()
  let hook_events := bases.map (fun base => ClassDefEvent.init_subclass_called base name)
  let update_events := bases.map (fun base => ClassDefEvent.subclasses_updated base name)
  let final_events := [.init_called name, .class_bound name]
  -- Update runtime
  let new_rt : PyRuntime := {
    classes := rt.classes ++ [cls],
    subclass_registry := fun parent => 
      if bases.contains parent 
      then rt.subclass_registry parent ++ [name]
      else rt.subclass_registry parent,
    init_subclass_calls := rt.init_subclass_calls ++ bases.map (fun base => (base, name))
  }
  (new_rt, events ++ hook_events ++ update_events ++ final_events)

/-!
## THEOREM: __init_subclass__ executes at definition time

This is PROVED from the semantics, not assumed.
The proof shows that init_subclass_called events are part of execute_class_statement's output.
-/

-- Helper functions for event lists
def initial_events (name : PyId) : List ClassDefEvent :=
  [.metacall_start name, .new_called name, .namespace_populated name]

def hook_events (name : PyId) (bases : List PyId) : List ClassDefEvent :=
  bases.map (fun base => ClassDefEvent.init_subclass_called base name)

def update_events (name : PyId) (bases : List PyId) : List ClassDefEvent :=
  bases.map (fun base => ClassDefEvent.subclasses_updated base name)

def final_events (name : PyId) : List ClassDefEvent :=
  [ClassDefEvent.init_called name, ClassDefEvent.class_bound name]

-- All events combined (using explicit concatenation order)
def class_def_all_events (name : PyId) (bases : List PyId) : List ClassDefEvent :=
  initial_events name ++ hook_events name bases ++ update_events name bases ++ final_events name

theorem execute_produces_events (rt : PyRuntime) (name : PyId) (bases : List PyId)
    (attrs methods : List (String × String)) :
    (execute_class_statement rt name bases attrs methods).2 = class_def_all_events name bases := by
  rfl

-- Helper for the proof: hook_events portion
def hook_events_portion (name : PyId) (bases : List PyId) : List ClassDefEvent :=
  bases.map (fun base => ClassDefEvent.init_subclass_called base name)

theorem hook_in_portion (name : PyId) (bases : List PyId) (parent : PyId) (h : parent ∈ bases) :
    ClassDefEvent.init_subclass_called parent name ∈ hook_events_portion name bases := by
  simp only [hook_events_portion, List.mem_map]
  exact ⟨parent, h, rfl⟩

-- hook_events_portion = hook_events (by definition)
theorem hook_portion_eq (name : PyId) (bases : List PyId) :
    hook_events_portion name bases = hook_events name bases := rfl

-- The class_def_all_events contains the hook_events
-- Structure: initial ++ hook ++ update ++ final
-- After simp: ((in initial ∨ in hook) ∨ in update) ∨ in final
-- We want to prove via the "in hook" branch
theorem hook_in_all (name : PyId) (bases : List PyId) :
    ∀ e ∈ hook_events name bases, e ∈ class_def_all_events name bases := by
  intro e he
  unfold class_def_all_events
  simp only [List.mem_append, hook_events, update_events, initial_events, final_events]
  -- Goal: ((in initial ∨ in hook) ∨ in update) ∨ in final
  left   -- ((in initial ∨ in hook) ∨ in update)
  left   -- (in initial ∨ in hook)
  right  -- in hook
  simp only [hook_events] at he
  exact he

theorem portion_subset_all (name : PyId) (bases : List PyId) :
    ∀ e ∈ hook_events_portion name bases, e ∈ class_def_all_events name bases := by
  rw [hook_portion_eq]
  exact hook_in_all name bases

theorem hook_event_in_all_events (name : PyId) (bases : List PyId) (parent : PyId) (h : parent ∈ bases) :
    ClassDefEvent.init_subclass_called parent name ∈ class_def_all_events name bases := by
  apply portion_subset_all
  exact hook_in_portion name bases parent h

theorem init_subclass_in_class_definition (rt : PyRuntime) (name : PyId) (bases : List PyId)
    (attrs : List (String × String)) (methods : List (String × String))
    (parent : PyId) (h : parent ∈ bases) :
    ClassDefEvent.init_subclass_called parent name ∈
    (execute_class_statement rt name bases attrs methods).2 := by
  rw [execute_produces_events]
  exact hook_event_in_all_events name bases parent h

-- Corollary: __init_subclass__ is a definition-time hook
theorem init_subclass_is_definition_hook :
    ∀ rt name bases attrs methods parent,
    parent ∈ bases →
    ∃ events, (execute_class_statement rt name bases attrs methods).2 = events ∧
    ClassDefEvent.init_subclass_called parent name ∈ events := by
  intro rt name bases attrs methods parent h
  exact ⟨(execute_class_statement rt name bases attrs methods).2, rfl, 
         init_subclass_in_class_definition rt name bases attrs methods parent h⟩

/-!
## THEOREM: __subclasses__() is updated at definition time

The subclass_registry is updated during execute_class_statement.
This means __subclasses__() returns accurate data immediately after definition.
-/

theorem subclasses_updated_at_definition (rt : PyRuntime) (name : PyId) (bases : List PyId)
    (attrs : List (String × String)) (methods : List (String × String))
    (parent : PyId) (h : parent ∈ bases) :
    name ∈ (execute_class_statement rt name bases attrs methods).1.subclass_registry parent := by
  simp [execute_class_statement]
  simp [h]

-- The subclasses query returns the updated list
def query_subclasses (rt : PyRuntime) (cls : PyId) : List PyId :=
  rt.subclass_registry cls

theorem subclasses_query_reflects_definition (rt : PyRuntime) (name : PyId) (bases : List PyId)
    (attrs methods) (parent : PyId) (h : parent ∈ bases) :
    let rt' := (execute_class_statement rt name bases attrs methods).1
    name ∈ query_subclasses rt' parent := by
  simp [query_subclasses]
  exact subclasses_updated_at_definition rt name bases attrs methods parent h

/-!
## Capability Predicates (Operational)

These connect the concrete Python semantics to the abstract SSOT requirements.
-/

def HasDefinitionHooks : Prop :=
  ∀ rt name bases attrs methods parent,
    parent ∈ bases →
    ClassDefEvent.init_subclass_called parent name ∈
      (execute_class_statement rt name bases attrs methods).2

def HasIntrospection : Prop :=
  ∀ rt name bases attrs methods parent,
    parent ∈ bases →
    let rt' := (execute_class_statement rt name bases attrs methods).1
    name ∈ query_subclasses rt' parent

theorem python_has_hooks : HasDefinitionHooks := by
  intro rt name bases attrs methods parent h
  exact init_subclass_in_class_definition rt name bases attrs methods parent h

theorem python_has_introspection : HasIntrospection := by
  intro rt name bases attrs methods parent h
  exact subclasses_query_reflects_definition rt name bases attrs methods parent h

end Python
