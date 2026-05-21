import { useParams } from "react-router-dom";

export function AgentDetailPage() {
  const { id } = useParams<{ id: string }>();
  return <h1>Agent {id}</h1>;
}
