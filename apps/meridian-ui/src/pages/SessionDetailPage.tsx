import { useParams } from "react-router-dom";

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  return <h1>Session {id}</h1>;
}
