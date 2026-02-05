import { LoaderCircleIcon } from "lucide-react";

export default function Home() {
    return (
        <div className="w-dvw h-dvh flex justify-center items-center">
            <LoaderCircleIcon
                size={24}
                className="stroke-gray-800 animate-spin"
            />
        </div>
    );
}
